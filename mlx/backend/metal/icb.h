// Copyright © 2026 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <array>
#include <cstdint>
#include <tuple>
#include <unordered_map>
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

  // Read-only view into the used portion of the inline-bytes arena.
  // For diagnostics and parity tests only — returned pointer is valid
  // for the recorder's lifetime. Returns `nullptr` if the arena wasn't
  // allocated (bytes_arena_cap_ == 0).
  const void* bytes_arena_ptr() const {
    return bytes_arena_ ? bytes_arena_->contents() : nullptr;
  }

  // ─────────────────────────────────────────────────────────────────────
  // Named binding tags (for replay-with-overrides)
  // ─────────────────────────────────────────────────────────────────────
  //
  // During recording, callers may associate a caller-chosen `name_id`
  // with an MTLBuffer that flows through the recorded dispatches. At
  // replay time, a different MTLBuffer can be substituted at every slot
  // that was originally bound with the tagged buffer. This is how a
  // per-layer ICB can be re-used across decode steps: each step rebinds
  // the layer's mutable inputs (current hidden state, KV cache pointers,
  // output slot) without re-encoding the dispatch sequence.

  // Concrete binding location in the recorder. Populated as bindings
  // are observed. `offset` preserves the original offset at record
  // time so callers can override the buffer while keeping the offset
  // (typical case: same tensor, different step).
  struct TagLocation {
    size_t segment_idx;
    size_t command_in_segment;
    int slot;
    int64_t offset;
  };

  // Associate `name_id` with every slot currently (or subsequently)
  // bound with `buf`. Behavior:
  //   1. Scans every already-recorded binding and registers a
  //      TagLocation for each match.
  //   2. Registers a pending tag so that any *future* set_kernel_buffer
  //      call with the same `buf` (for the lifetime of this recorder,
  //      until finalize()) also produces a TagLocation.
  //
  // Returns the number of immediate matches found (step 1 only).
  // Rationale: callers often tag before the layer they care about runs
  // (e.g. tag the hidden-state input, then execute the layer). In that
  // case step 1 returns 0 and the pending mechanism supplies the tags
  // as bindings happen.
  size_t tag_binding(uint32_t name_id, const MTL::Buffer* buf);

  // All recorded TagLocations for `name_id`. Empty vector if unknown.
  const std::vector<TagLocation>& tags_for(uint32_t name_id) const;

  // Replay with per-name buffer overrides. For each (name_id, buffer,
  // offset) entry:
  //   - Look up every TagLocation registered under `name_id`.
  //   - Rewrite the corresponding command's slot in the materialized
  //     `MTL::IndirectCommandBuffer` via `setKernelBuffer`, using the
  //     supplied offset.
  //   - Ensure the override buffer is covered by `useResource` on the
  //     live encoder before the segment's `executeCommandsInBuffer`.
  //
  // Any tag names not present in `overrides` keep their recorded
  // binding. The ICB's commands are mutated in place — a subsequent
  // replay() (no overrides) will observe the last overrides written.
  // Callers typically replay_with_overrides every step.
  void replay_with_overrides(
      MTL::ComputeCommandEncoder* enc,
      const std::vector<
          std::tuple<uint32_t, const MTL::Buffer*, int64_t>>& overrides);

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

  // Union of every segment's `resource_set`, built at finalize().
  // Used at replay time to call `useResource` once per unique
  // buffer on the live encoder (Metal's useResource is encoder-
  // scoped, not per-dispatch, so per-segment iteration is
  // redundant and paid many thousand API calls per decode step on
  // models like GPT-OSS 20B with ~650 segments).
  std::vector<const MTL::Buffer*> all_resources_;

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

  // Named-binding tag storage. Populated by tag_binding and maintained
  // by end_command (which scans cur_ bindings for pending matches
  // before cur_ lands in a segment).
  std::unordered_map<uint32_t, std::vector<TagLocation>> tags_;
  std::unordered_map<const MTL::Buffer*, std::vector<uint32_t>>
      pending_tags_by_buffer_;
  static const std::vector<TagLocation> kEmptyTagLocations_;
};

} // namespace mlx::core::metal
