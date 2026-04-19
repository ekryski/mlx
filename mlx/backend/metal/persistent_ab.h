// Copyright © 2026 Apple Inc.

#pragma once

#include <vector>

#include "mlx/api.h"
#include "mlx/backend/metal/argument_buffer.h"

namespace mlx::core::metal {

class Device;

// An ArgumentBuffer whose lifetime is caller-managed, rather than
// tied to a single command buffer's completion.
//
// Motivation — Indirect Command Buffer replay
// -------------------------------------------
// The ICB recorder captures a `setBuffer(ab_mtl, 0)` bind at record
// time. For the replay to be correct across decode steps, the AB's
// backing MTLBuffer must remain addressable AND its contents must
// reflect the current step's state. A transient ArgumentBuffer —
// one released via `CommandEncoder::add_temporary_object` when the
// recording command buffer completes — gets freed before the next
// replay runs. `PersistentAb` closes that gap: the owning caller
// keeps the handle alive across replays; the underlying MTLBuffer
// address is stable; the caller rewrites its contents between
// replays by calling the typed setters on each step.
//
// Design notes
// ------------
// - Non-copyable, non-movable. Hold via `std::shared_ptr` or
//   `std::unique_ptr` where handle semantics matter. This keeps
//   the ownership story simple: exactly one owner, one MTLBuffer.
// - Wraps `ArgumentBuffer` by value and forwards the typed setters.
//   All contents-update behavior (slot layout, GPU-visible bytes,
//   packing rules) is inherited verbatim from ArgumentBuffer; the
//   only behavioral difference is lifetime.
// - Underlying MTLBuffer is pooled via `ArgumentBuffer`'s pool (same
//   pool as transient ABs). On destruction the buffer is returned to
//   the pool.
class MLX_API PersistentAb {
 public:
  // Allocates the backing MTLBuffer via the shared BufferPool. The
  // slot ordering and kinds MUST match the kernel-side struct this
  // AB will be bound to. The slot `byte_offset` on each entry is
  // ignored on input and populated by ArgumentBuffer's packing.
  PersistentAb(Device& d, std::vector<ArgumentBuffer::Slot> layout);
  ~PersistentAb();

  PersistentAb(const PersistentAb&) = delete;
  PersistentAb& operator=(const PersistentAb&) = delete;
  PersistentAb(PersistentAb&&) = delete;
  PersistentAb& operator=(PersistentAb&&) = delete;

  // Typed setters — mirror ArgumentBuffer. Mismatched slot indices
  // or kinds throw with a clear message. Each setter performs an
  // aligned memcpy into the shared-storage contents.
  void set_scalar32(int slot, uint32_t value);
  void set_scalar64(int slot, uint64_t value);
  void set_float32(int slot, float value);
  void set_buffer_ptr(int slot, const MTL::Buffer* buf, int64_t offset);
  void set_bytes(int slot, const void* data, size_t size);

  // Raw MTLBuffer for binding at a compute encoder slot or for
  // recording inside an ICB. The pointer is stable for the
  // PersistentAb's lifetime.
  MTL::Buffer* mtl_buffer() const;

  // Layout introspection. Useful for callers that need to validate
  // slot byte-offsets against a kernel-side struct at test time.
  const std::vector<ArgumentBuffer::Slot>& layout() const;
  size_t size_bytes() const;

 private:
  ArgumentBuffer ab_;
};

} // namespace mlx::core::metal

namespace mlx::core {

// Push a caller-owned PersistentAb onto the thread-local handoff
// queue for upcoming Gather::eval_gpu invocations that enter the
// `gather_front_ab` AB path (indexing.cpp). FIFO: the next matching
// gather consumes the front handle; additional gathers consume
// subsequent handles. When the queue is drained, further gathers
// fall back to transient ABs.
//
// Used by the decode-loop ICB orchestrator to keep the embedding
// gather's indices-pointer mutable between replays. Pushed in
// sequence because QuantizedEmbedding issues multiple back-to-back
// gathers (`weight[x]`, `scales[x]`, `biases[x]`) per lookup.
MLX_API void push_next_gather_front_persistent_ab(
    metal::PersistentAb* handle);
MLX_API metal::PersistentAb* consume_next_gather_front_persistent_ab();
MLX_API void clear_next_gather_front_persistent_abs();
MLX_API size_t pending_gather_front_persistent_abs();

} // namespace mlx::core
