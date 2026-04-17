// Copyright © 2026 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <array>
#include <unordered_set>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::metal {

class Device;

// Captures a stable sequence of compute dispatches as an
// MTLIndirectCommandBuffer so the sequence can be replayed on subsequent
// evaluations with substantially lower CPU encoding cost.
//
// Feasibility: see tests/icb_feasibility_tests.cpp — ~17x speedup on
// encoding 1500 trivial dispatches (M1 Max, macOS 26.2).
//
// Design notes:
//   - Bindings are captured per-slot into a compact command table and
//     materialized into the ICB on `finalize()`. This keeps the hot path
//     (record_* calls) allocation-free after construction.
//   - `MTLIndirectComputeCommand` has no `setBytes` — inline constants
//     spill into an arena-allocated shared-memory buffer during recording
//     and bind as a kernel buffer on replay.
//   - ICB execution uses `IndirectCommandTypeConcurrentDispatch`, so the
//     recorded sequence must be barrier-free. A caller that needs
//     barriers splits the work into multiple recorders and interleaves
//     barriers on the live encoder between replays. This is handled one
//     level up (by `CommandEncoder` integration), not here.
//   - The recorder holds references to every MTLBuffer and pipeline it
//     recorded so they cannot be freed while the ICB is alive. Replay
//     callers still need to `useResource` those buffers on the executing
//     command buffer — `replay()` does this automatically.
class MLX_API IndirectCommandRecorder {
 public:
  // Hard cap per Metal's compute ICB limits (tier-2 argument buffer GPUs
  // support 31 compute-kernel buffer bindings). We set the descriptor's
  // maxKernelBufferBindCount to this value.
  static constexpr int kMaxKernelBufferBindCount = 31;

  // Feature gate. Returns true on any GPU that supports MTLIndirectCommandBuffer
  // for compute with per-command pipeline binding. On Apple Silicon this is
  // effectively every M-series device, but we probe at runtime rather than
  // assuming.
  static bool is_supported(Device& d);

  // `max_commands` is the capacity of the underlying ICB. `bytes_arena_cap`
  // is the total pool available for spilled inline constants across all
  // commands.
  IndirectCommandRecorder(
      Device& d,
      size_t max_commands,
      size_t bytes_arena_cap = 64 * 1024);
  ~IndirectCommandRecorder();

  IndirectCommandRecorder(const IndirectCommandRecorder&) = delete;
  IndirectCommandRecorder& operator=(const IndirectCommandRecorder&) = delete;

  // Start a new command. `pipeline` must have been built with
  // setSupportIndirectCommandBuffers(true); passing a non-ICB pipeline
  // results in an invalid ICB at finalize time.
  void begin_command(MTL::ComputePipelineState* pipeline);

  // Bind a device buffer at `slot` for the current command. `slot` must
  // be in [0, kMaxKernelBufferBindCount).
  void set_kernel_buffer(const MTL::Buffer* buf, int64_t offset, int slot);

  // Spill `length` bytes into the arena and bind the resulting region at
  // `slot`. Returns false if the arena is exhausted; the caller should
  // then treat the recording as failed and fall back to direct dispatch.
  [[nodiscard]] bool set_bytes(const void* data, size_t length, int slot);

  // Finalize the current command. If `use_dispatch_threads` is true, the
  // command is emitted with concurrentDispatchThreads (i.e. `grid` is
  // total thread count); otherwise concurrentDispatchThreadgroups is used
  // (`grid` is in threadgroup units).
  void end_command(MTL::Size grid, MTL::Size group, bool use_dispatch_threads);

  // Number of commands recorded so far.
  size_t size() const {
    return next_cmd_;
  }

  // Whether the recorder has been finalized; replay is only legal after.
  bool finalized() const {
    return finalized_;
  }

  // Materialize the recorded commands into the ICB. Must be called before
  // `replay`. No further commands can be recorded after `finalize`.
  void finalize();

  // Replay the recorded sequence on `enc`. The caller must have a live
  // compute encoder. `replay` will:
  //   - Call `useResource` on every MTLBuffer the recording references,
  //     including the bytes-arena buffer itself.
  //   - Issue a single `executeCommandsInBuffer` for the full range.
  void replay(MTL::ComputeCommandEncoder* enc) const;

  // Access to the underlying ICB (for debugging / advanced integration).
  const MTL::IndirectCommandBuffer* icb() const {
    return icb_.get();
  }

  // Total bytes consumed from the inline-bytes arena so far. For telemetry.
  size_t bytes_arena_used() const {
    return bytes_offset_;
  }

 private:
  struct Binding {
    const MTL::Buffer* buffer = nullptr;
    int64_t offset = 0;
  };

  struct Command {
    MTL::ComputePipelineState* pipeline = nullptr;
    std::array<Binding, kMaxKernelBufferBindCount> bindings{};
    MTL::Size grid{0, 0, 0};
    MTL::Size group{0, 0, 0};
    bool use_dispatch_threads = false;
  };

  Device& device_;
  size_t max_commands_;
  size_t bytes_arena_cap_;

  NS::SharedPtr<MTL::IndirectCommandBuffer> icb_;

  std::vector<Command> commands_;
  Command cur_{};
  bool cur_active_ = false;
  size_t next_cmd_ = 0;

  // Shared-memory arena for spilled `setBytes` payloads. Shared so we can
  // memcpy into it from the CPU cheaply; each command that uses bytes
  // carves a suballocation and binds it at the appropriate slot.
  NS::SharedPtr<MTL::Buffer> bytes_arena_;
  size_t bytes_offset_ = 0;

  // Unique set of referenced buffers — used to drive `useResource` on
  // replay. Includes the bytes arena itself.
  std::unordered_set<const MTL::Buffer*> resource_set_;

  bool finalized_ = false;
};

} // namespace mlx::core::metal
