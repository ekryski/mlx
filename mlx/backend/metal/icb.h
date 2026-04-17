// Copyright © 2026 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <array>
#include <unordered_set>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::metal {

class Device;

// Captures a stable sequence of compute dispatches as one or more
// MTLIndirectCommandBuffer segments, separated by memory barriers. The
// captured sequence can be replayed on subsequent evaluations with
// substantially lower CPU encoding cost than direct dispatch.
//
// Feasibility: see tests/icb_feasibility_tests.cpp — ~17x speedup on
// encoding 1500 trivial dispatches (M1 Max, macOS 26.2).
//
// Design notes:
//   - `MTL::IndirectComputeCommand` cannot emit memory barriers, so the
//     recorder splits the recording into multiple ICB segments at every
//     caller-requested barrier boundary. Replay walks segments and emits
//     `memoryBarrier(BarrierScopeBuffers)` on the live encoder between
//     them.
//   - `MTL::IndirectComputeCommand` has no `setBytes` — inline constants
//     spill into a shared-memory arena during recording and bind as a
//     kernel buffer on replay. The arena is shared across all segments.
//   - Strong-references every MTLBuffer it binds so replay after mlx's
//     original array references go out of scope is safe.
class MLX_API IndirectCommandRecorder {
 public:
  // Hard cap per Metal's compute ICB limits (tier-2 argument buffer GPUs
  // support 31 compute-kernel buffer bindings).
  static constexpr int kMaxKernelBufferBindCount = 31;

  // Feature gate — effectively every M-series device.
  static bool is_supported(Device& d);

  // `max_commands_per_segment` is the capacity of each ICB segment.
  // `bytes_arena_cap` is the total pool shared across all segments for
  // spilled inline constants.
  IndirectCommandRecorder(
      Device& d,
      size_t max_commands_per_segment,
      size_t bytes_arena_cap = 64 * 1024);
  ~IndirectCommandRecorder();

  IndirectCommandRecorder(const IndirectCommandRecorder&) = delete;
  IndirectCommandRecorder& operator=(const IndirectCommandRecorder&) = delete;

  // Start a new command. `pipeline` must have been built with
  // setSupportIndirectCommandBuffers(true).
  void begin_command(MTL::ComputePipelineState* pipeline);

  // Bind a device buffer at `slot` for the current command.
  void set_kernel_buffer(const MTL::Buffer* buf, int64_t offset, int slot);

  // Spill `length` bytes into the arena and bind the resulting region at
  // `slot`. Returns false if the arena is exhausted.
  [[nodiscard]] bool set_bytes(const void* data, size_t length, int slot);

  // Set threadgroup memory length at `slot` for the current command.
  void set_threadgroup_memory(size_t length, int slot);

  // Finalize the current command.
  void end_command(MTL::Size grid, MTL::Size group, bool use_dispatch_threads);

  // Close the current segment (allocating and populating its ICB), open
  // a new segment for subsequent commands. Replay inserts a memory
  // barrier between consecutive segments. No-op if the current segment
  // is empty (no wasted barriers). Must not be called while a command
  // is partially built.
  void split_segment();

  // Total number of commands across all segments.
  size_t size() const {
    return total_commands_;
  }

  // Number of segments (ICBs) recorded. Always ≥ 1 after first command.
  size_t num_segments() const {
    return segments_.size();
  }

  bool finalized() const {
    return finalized_;
  }

  // Materialize the final (open) segment's ICB. Must be called before
  // replay. No further record_* calls after this.
  void finalize();

  // Replay the full recorded sequence on `enc`. For each segment:
  //   - calls `useResource` on every referenced buffer, and
  //   - issues `executeCommandsInBuffer` for the segment's range.
  // Between consecutive segments, emits a memory barrier on `enc`.
  void replay(MTL::ComputeCommandEncoder* enc) const;

  // Total bytes consumed from the inline-bytes arena.
  size_t bytes_arena_used() const {
    return bytes_offset_;
  }

 private:
  struct Binding {
    const MTL::Buffer* buffer = nullptr;
    int64_t offset = 0;
  };

  static constexpr int kMaxThreadgroupMemorySlots = 8;

  struct ThreadgroupMem {
    int slot = -1;
    size_t length = 0;
  };

  struct Command {
    MTL::ComputePipelineState* pipeline = nullptr;
    std::array<Binding, kMaxKernelBufferBindCount> bindings{};
    std::array<ThreadgroupMem, kMaxThreadgroupMemorySlots> threadgroup_mem{};
    int threadgroup_mem_count = 0;
    MTL::Size grid{0, 0, 0};
    MTL::Size group{0, 0, 0};
    bool use_dispatch_threads = false;
  };

  struct Segment {
    NS::SharedPtr<MTL::IndirectCommandBuffer> icb;
    std::vector<Command> commands;
    std::unordered_set<const MTL::Buffer*> resource_set;
  };

  // Builds the descriptor and allocates the ICB for a segment, given its
  // known command count.
  NS::SharedPtr<MTL::IndirectCommandBuffer> allocate_icb_(size_t max_commands);

  // Materialize `commands` into `icb` (set pipeline, bindings, threadgroup
  // mem, dispatch) for one segment.
  static void materialize_icb_(
      const std::vector<Command>& commands,
      MTL::IndirectCommandBuffer* icb);

  // Add a buffer to the current segment's resource_set and retain if new.
  void track_resource_(const MTL::Buffer* buf);

  Device& device_;
  size_t max_commands_per_segment_;
  size_t bytes_arena_cap_;

  std::vector<Segment> segments_;

  // The command currently being built.
  Command cur_{};
  bool cur_active_ = false;
  size_t total_commands_ = 0;

  // Shared-memory arena for spilled setBytes payloads. Shared across all
  // segments in this recorder.
  NS::SharedPtr<MTL::Buffer> bytes_arena_;
  size_t bytes_offset_ = 0;

  // Strong refs to every unique MTLBuffer we bound, so backing storage
  // outlives any mlx::core::array that originally held it.
  std::vector<NS::SharedPtr<MTL::Buffer>> retained_buffers_;
  std::unordered_set<const MTL::Buffer*> all_retained_;

  bool finalized_ = false;
};

} // namespace mlx::core::metal
