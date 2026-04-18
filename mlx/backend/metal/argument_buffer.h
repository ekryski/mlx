// Copyright © 2026 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::metal {

class Device;

// A pre-sized Metal buffer with a typed, positional slot layout. Holds
// the resource pointers and inline scalars a compute kernel would
// otherwise receive via many individual `setBuffer` / `setBytes` calls
// — packing them into one shared-storage buffer so the entire argument
// state can be bound with a single `setBuffer` (or referenced from an
// ICB) and updated in place between dispatches.
//
// Purpose: argument buffers are the Metal-recommended pairing with
// Indirect Command Buffers. They let the CPU update per-step kernel
// inputs (shape scalars, offsets, fresh buffer pointers) without
// re-encoding the dispatch sequence. See
// benchmarks/notes/argument-buffers-adoption-plan-2026-04-17.md for
// the wider rollout strategy.
//
// Layout is declared at construction time. Each `Slot` records the
// byte offset into the buffer and the type of value written there.
// Slot indices are stable across `set_*` calls — callers store the
// index once at build time and reference it by position.
//
// Storage mode: `MTL::ResourceStorageModeShared` so CPU writes land in
// the same memory the GPU reads. Sizes are small (typically 32–256 B
// per primitive) so the storage choice has no measurable cost.
//
// Thread-safety: one `ArgumentBuffer` per primitive invocation. The
// class is not internally synchronized.
class MLX_API ArgumentBuffer {
 public:
  // Supported slot payloads. The layout is tight (natural alignment
  // per kind); slots are packed in declaration order by the ctor.
  struct Slot {
    enum class Kind : uint8_t {
      // 32-bit unsigned integer (shape dims, strides, offsets).
      Scalar32,
      // 64-bit unsigned integer (large offsets, kernel-internal IDs).
      Scalar64,
      // 32-bit float (scales, eps, etc.).
      Float32,
      // Pointer to a device buffer + byte offset. On GPU this slot
      // reads as a 16-byte struct: `{ uint64_t addr; uint64_t offset; }`.
      BufferPtrOffset,
    };
    Kind kind;
    // Byte offset into the argument buffer where this slot lives.
    // Populated by the constructor from `layout.size()` in
    // declaration order.
    size_t byte_offset = 0;
    // Optional human-readable name for diagnostics / tests. Empty for
    // production use.
    std::string name;
  };

  // Construct with a declarative layout. Slot `i` in the returned
  // layout corresponds to slot index `i` in the `set_*` calls. The
  // underlying MTL::Buffer is allocated immediately with enough room
  // for every slot.
  //
  // `slots` layout payloads (kind only — `byte_offset` is ignored on
  // input and populated by the ctor).
  ArgumentBuffer(Device& d, std::vector<Slot> slots);
  ~ArgumentBuffer();

  ArgumentBuffer(const ArgumentBuffer&) = delete;
  ArgumentBuffer& operator=(const ArgumentBuffer&) = delete;

  // Write a 32-bit unsigned integer to `slot`. `slot` must reference
  // a `Scalar32` slot in the layout.
  void set_scalar32(int slot, uint32_t value);

  // Write a 64-bit unsigned integer to `slot`. `slot` must reference
  // a `Scalar64` slot in the layout.
  void set_scalar64(int slot, uint64_t value);

  // Write a 32-bit float to `slot`. `slot` must reference a
  // `Float32` slot in the layout.
  void set_float32(int slot, float value);

  // Write a `(buffer_addr, offset)` pair to `slot`. `slot` must
  // reference a `BufferPtrOffset` slot in the layout. The buffer is
  // NOT retained by the argument buffer — the caller owns the buffer
  // lifetime. `offset` is in bytes.
  void set_buffer_ptr(int slot, const MTL::Buffer* buf, int64_t offset);

  // The underlying device buffer. Bind this at a single kernel-arg
  // slot (via setBuffer) or reference it from an ICB's command. The
  // pointer is stable for the lifetime of the ArgumentBuffer.
  MTL::Buffer* mtl_buffer() const {
    return buffer_.get();
  }

  // Total size in bytes of the argument buffer (layout-driven).
  size_t size_bytes() const {
    return size_;
  }

  // Layout accessors — expose the finalized slot list so callers can
  // validate indices and retrieve byte offsets from test code.
  const std::vector<Slot>& layout() const {
    return layout_;
  }

 private:
  // Raw contents pointer for in-place CPU writes.
  uint8_t* contents_() {
    return static_cast<uint8_t*>(buffer_->contents());
  }

  // Byte layout of one BufferPtrOffset slot on GPU.
  struct BufferPtrOffsetEntry {
    uint64_t addr;
    uint64_t offset;
  };

  std::vector<Slot> layout_;
  NS::SharedPtr<MTL::Buffer> buffer_;
  size_t size_ = 0;
};

} // namespace mlx::core::metal
