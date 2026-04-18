// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/backend/metal/resident.h"
#include "mlx/device.h"

namespace mlx::core::metal {

using MTLFCList =
    std::vector<std::tuple<const void*, MTL::DataType, NS::UInteger>>;

class Device;
class IndirectCommandRecorder;

class MLX_API CommandEncoder {
 public:
  CommandEncoder(Device& d, int index, ResidencySet& residency_set);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  struct ConcurrentContext {
    ConcurrentContext(CommandEncoder& enc) : enc(enc) {
      enc.concurrent_ = true;
    }
    ~ConcurrentContext() {
      enc.concurrent_ = false;
      enc.prev_outputs_.insert(
          enc.concurrent_outputs_.begin(), enc.concurrent_outputs_.end());
      enc.concurrent_outputs_.clear();
    }

   private:
    CommandEncoder& enc;
  };

  void set_buffer(const MTL::Buffer* buf, int idx, int64_t offset = 0);
  void set_input_array(const array& a, int idx, int64_t offset = 0);
  void set_output_array(array& a, int idx, int64_t offset = 0);
  // Dependency-only registration — like `set_input_array` but without
  // issuing a `setBuffer` on the live encoder. Use when the buffer is
  // consumed indirectly (e.g. via an ArgumentBuffer) so barrier
  // tracking and cross-encoder fencing still see it.
  void register_input_array(const array& a);
  void register_output_array(const array& a);

  void add_temporary(array arr);
  void add_temporaries(std::vector<array> arrays);

  // Retain an arbitrary heap-allocated helper object (e.g. an
  // ArgumentBuffer) until the GPU finishes the current command buffer.
  // Released in the command-buffer completion handler alongside
  // `temporaries_`.
  void add_temporary_object(std::shared_ptr<void> obj);

  void dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims);
  void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
  void maybeInsertBarrier();

  void set_compute_pipeline_state(MTL::ComputePipelineState* kernel);

  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, size_t nelems, int idx) {
    set_bytes_raw(
        vec.data(), nelems * sizeof(typename Vec::value_type), idx);
  }
  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, int idx) {
    return set_vector_bytes(vec, vec.size(), idx);
  }

  template <typename T>
  void set_bytes(const T* v, int n, int idx) {
    set_bytes_raw(v, n * sizeof(T), idx);
  }

  template <typename T>
  void set_bytes(const T& v, int idx) {
    set_bytes_raw(&v, sizeof(T), idx);
  }

  void set_threadgroup_memory_length(size_t length, int idx);

  ConcurrentContext start_concurrent() {
    return ConcurrentContext(*this);
  }

  void barrier();
  void end_encoding();
  bool needs_commit() const;
  void commit();
  void synchronize();

  // Dispatch counter — cumulative count of dispatchThreadgroups /
  // dispatchThreads calls since the last reset_dispatch_counter(). Used to
  // audit per-token dispatch counts for ICB feasibility studies and to
  // quantify the CPU-encoding optimization target.
  void reset_dispatch_counter();
  uint64_t total_dispatches() const;

  // Kernel-name logging — when started, every call to
  // `set_compute_pipeline_state` followed by a `dispatch_*` appends the
  // pipeline state's `label()` to an in-memory log. Enables the
  // ICB-stability audit (compare kernel-name sequences across tokens
  // — identical sequences mean the dispatch list is stable and the ICB
  // encode-once/execute-many path is viable).
  static void start_kernel_log();
  static void stop_kernel_log();
  static size_t kernel_log_size();
  // Get the label at index i. Returns nullptr if out of range. The
  // returned C-string is valid until the next `start_kernel_log()`.
  static const char* kernel_log_at(size_t i);

  // Indirect Command Buffer recording.
  //
  // While recording is active, calls to `set_compute_pipeline_state`,
  // `set_input_array` / `set_output_array` / `set_buffer`, `set_bytes`,
  // and `dispatch_*` are routed into a side-channel `IndirectCommandRecorder`
  // instead of emitting live commands. `maybeInsertBarrier` is a no-op
  // while recording — the caller is responsible for ensuring the recorded
  // sequence is barrier-independent (see docs on IndirectCommandRecorder).
  //
  // Finalize via `end_icb_recording`, which returns the recorder. The
  // caller owns the recorder and can replay it later via `replay_icb` on
  // any CommandEncoder bound to the same device.
  void begin_icb_recording(size_t max_commands, size_t bytes_arena_cap = 64 * 1024);
  std::unique_ptr<IndirectCommandRecorder> end_icb_recording();
  // Cancel the current recording session and discard any captured work.
  // Safe to call whether or not `recording_` is currently true. Use this
  // when a recording block throws and the caller wants a clean slate
  // before reissuing work.
  void abort_icb_recording();
  void replay_icb(const IndirectCommandRecorder& recorder);

  // Named-binding tag during recording. Must be called while a
  // recording session is active; associates `name_id` with the
  // underlying MTLBuffer of `a`. Any already-recorded command that
  // binds that buffer (at any slot) is tagged immediately; any
  // subsequent bind of the same buffer under this recorder is also
  // tagged at end_command time. See
  // `IndirectCommandRecorder::tag_binding` for semantics.
  void tag_binding(uint32_t name_id, const array& a);
  void tag_binding(uint32_t name_id, const MTL::Buffer* buf);

  // Replay a recorded IndirectCommandRecorder with per-name buffer
  // overrides. Each override triple is (name_id, override_buffer,
  // override_offset). See
  // `IndirectCommandRecorder::replay_with_overrides` for semantics —
  // in particular, override writes mutate the recorder's ICBs in place.
  void replay_icb_with_overrides(
      IndirectCommandRecorder& recorder,
      const std::vector<
          std::tuple<uint32_t, const MTL::Buffer*, int64_t>>& overrides);

  bool is_recording() const {
    return recording_;
  }
  // Number of set_input_array / set_buffer calls that were silently
  // skipped during the most recent recording session because no
  // pipeline command was in progress. High counts indicate primitives
  // relying on sticky bindings across dispatches — an ICB-incompatible
  // pattern. Reset to 0 on begin_icb_recording.
  size_t icb_skipped_set_input_count() const {
    return icb_skipped_set_input_;
  }

  MTL::CommandQueue* get_command_queue() const {
    return queue_.get();
  }
  MTL::CommandBuffer* get_command_buffer() const {
    return buffer_.get();
  }

 private:
  MTL::ComputeCommandEncoder* get_command_encoder();
  void set_bytes_raw(const void* data, size_t length, int idx);
  // Internal: force recording state back to idle (clear flag + recorder +
  // pending command). Used by throw sites so exceptions don't leave the
  // encoder in a half-recording state.
  void abort_icb_recording_();

  Device& device_;
  bool exiting_{false};

  // ICB recording state. When `recording_` is true, the encoder routes
  // dispatch + binding calls into `active_recorder_` instead of emitting
  // live Metal commands.
  bool recording_{false};
  std::unique_ptr<IndirectCommandRecorder> active_recorder_;
  // Staged dispatch state: set by `set_compute_pipeline_state` and closed
  // out by `dispatch_*`. Only meaningful while recording.
  bool has_pending_command_{false};
  // Diagnostic counter — how many set_input_array / set_buffer calls
  // we ignored because no pipeline was bound. Exposes the correctness
  // gap for ICB replay (each skipped bind may mean a dispatch runs
  // with missing arguments).
  size_t icb_skipped_set_input_{0};
  // Diagnostic counter — how many dispatch_threadgroups / dispatch_threads
  // calls reached the encoder during the active recording. Compare with
  // the ICB's recorded command count: a gap means some dispatches were
  // begun but never finalized (e.g. set_compute_pipeline_state called
  // twice in a row clobbered `cur_`) or the primitive called dispatch
  // via a path we don't route through `recording_`.
  size_t icb_dispatch_calls_{0};

  // Buffer that stores encoded commands.
  NS::SharedPtr<MTL::CommandQueue> queue_;
  NS::SharedPtr<MTL::CommandBuffer> buffer_;
  int buffer_ops_{0};
  size_t buffer_sizes_{0};

  // Encoder for issuing GPU commands.
  // The members are used within a single ComputeCommandEncoder and will be
  // reset after calling end_encoding().
  NS::SharedPtr<MTL::ComputeCommandEncoder> encoder_;
  NS::SharedPtr<MTL::Fence> fence_;
  bool needs_barrier_{false};
  bool concurrent_{false};
  std::vector<array> temporaries_;
  // Heap-allocated helper objects kept alive for the life of the
  // current command buffer (released in the completion handler).
  std::vector<std::shared_ptr<void>> temporary_objects_;
  std::unordered_set<MTL::Resource*> prev_outputs_;
  std::unordered_set<MTL::Resource*> next_outputs_;
  std::unordered_set<MTL::Resource*> concurrent_outputs_;
  std::unordered_set<const void*> all_inputs_;
  std::unordered_set<const void*> all_outputs_;

  // A map of prior command encoder outputs to their corresponding fence.
  std::unordered_map<const void*, NS::SharedPtr<MTL::Fence>> prev_ce_outputs_;
  std::mutex outputs_mtx_;
};

class MLX_API Device {
 public:
  Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device();

  MTL::Device* mtl_device() {
    return device_.get();
  };

  const std::string& get_architecture() const {
    return arch_;
  }
  int get_architecture_gen() const {
    return arch_gen_;
  }
  std::tuple<int, int> get_max_ops_mb_per_buffer() const {
    return std::make_tuple(max_ops_per_buffer_, max_mb_per_buffer_);
  }

  MTL::Library* get_library(
      const std::string& name,
      const std::string& path = "");

  MTL::Library* get_library(
      const std::string& name,
      const std::function<std::string(void)>& builder);

  void clear_library(const std::string& name);

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  ResidencySet& residency_set() {
    return residency_set_;
  }

 private:
  NS::SharedPtr<MTL::Library> build_library_(const std::string& source_string);

  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      MTL::Library* mtl_lib);
  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      const std::string& specialized_name,
      const MTLFCList& func_consts,
      MTL::Library* mtl_lib);

  NS::SharedPtr<MTL::LinkedFunctions> get_linked_functions_(
      const std::vector<MTL::Function*>& funcs);

  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function);
  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function,
      const MTL::LinkedFunctions* linked_functions);

  MTL::ComputePipelineState* get_kernel_(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name,
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  NS::SharedPtr<MTL::Device> device_;
  ResidencySet residency_set_;

  std::shared_mutex kernel_mtx_;
  std::shared_mutex library_mtx_;
  std::unordered_map<std::string, NS::SharedPtr<MTL::Library>> library_map_;
  NS::SharedPtr<MTL::Library> default_library_;
  std::unordered_map<
      MTL::Library*,
      std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>>
      library_kernels_;
  std::string arch_;
  int arch_gen_;
  int max_ops_per_buffer_;
  int max_mb_per_buffer_;
};

MLX_API Device& device(mlx::core::Device);
MLX_API CommandEncoder& get_command_encoder(Stream s);

/// Register a name for a compute pipeline state. MLX calls this from
/// `Device::get_kernel_` at pipeline-build time so the CommandEncoder's
/// kernel log can resolve a pipeline pointer back to its
/// Metal-function-name string at dispatch time. Safe to call concurrently.
MLX_API void register_kernel_name(
    const MTL::ComputePipelineState* pipeline, const std::string& name);

/// Look up a previously-registered kernel name. Returns "" if the
/// pipeline wasn't registered (e.g., kernels built outside the standard
/// `get_kernel_` path).
MLX_API std::string kernel_name_for(const MTL::ComputePipelineState* pipeline);

std::unordered_map<int, CommandEncoder>& get_command_encoders();
NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool();

/// Programmatically enable or disable compilation of compute pipelines
/// with `supportIndirectCommandBuffers = true`. Affects subsequent
/// `Device::get_kernel_` calls only; pipelines already compiled are
/// untouched. Overrides the MLX_METAL_ICB env-var setting. Intended
/// for tests and benchmarks that need ICB capture without relying on
/// env-var ordering.
MLX_API void set_icb_pipeline_support(bool enabled);

bool is_nax_available();

} // namespace mlx::core::metal
