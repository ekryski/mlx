// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/argument_buffer.h"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

namespace {

// Process-wide pool of shared-storage MTL::Buffers, keyed by byte
// size. Acquired at ArgumentBuffer construction, returned on
// destruction.
//
// Motivation: per-call `MTL::Device::newBuffer` on a small (<1 KiB)
// shared buffer is expensive enough that at ~300 dispatches/step
// (qmv during Gemma4-E2B decode) it dominates the AB path's cost
// and turns a 1.975x encoding-cost win into a wall-time regression.
// Pooling drops the ctor hot path to a mutex + vector pop.
//
// Lifetime is tied to the `ArgumentBuffer` object; callers retain
// ABs via `CommandEncoder::add_temporary_object` until the owning
// command buffer completes on the GPU, so buffers are only returned
// to the pool after the GPU finishes reading them.
class BufferPool {
 public:
  NS::SharedPtr<MTL::Buffer> acquire(Device& d, size_t size) {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      auto it = free_list_.find(size);
      if (it != free_list_.end() && !it->second.empty()) {
        auto buf = std::move(it->second.back());
        it->second.pop_back();
        return buf;
      }
    }
    auto* dev = d.mtl_device();
    if (!dev) {
      throw std::runtime_error(
          "[metal::ArgumentBuffer::BufferPool] Metal device unavailable");
    }
    auto buf = NS::TransferPtr(
        dev->newBuffer(size, MTL::ResourceStorageModeShared));
    if (!buf) {
      throw std::runtime_error(
          "[metal::ArgumentBuffer::BufferPool] failed to allocate "
          "shared buffer of " + std::to_string(size) + " bytes");
    }
    return buf;
  }

  void release(size_t size, NS::SharedPtr<MTL::Buffer> buf) {
    std::lock_guard<std::mutex> lk(mtx_);
    free_list_[size].push_back(std::move(buf));
  }

 private:
  std::mutex mtx_;
  std::unordered_map<size_t, std::vector<NS::SharedPtr<MTL::Buffer>>>
      free_list_;
};

BufferPool& buffer_pool() {
  static BufferPool pool;
  return pool;
}

// Natural size of each slot kind in bytes. Layout is packed in
// declaration order with each slot aligned to its own size (matches
// Metal's default packing rules for these scalar types).
size_t slot_size(ArgumentBuffer::Slot::Kind k) {
  switch (k) {
    case ArgumentBuffer::Slot::Kind::Scalar32:
    case ArgumentBuffer::Slot::Kind::Float32:
      return 4;
    case ArgumentBuffer::Slot::Kind::Scalar64:
      return 8;
    case ArgumentBuffer::Slot::Kind::BufferPtrOffset:
      return 16; // uint64 addr + uint64 offset
  }
  // Unreachable — all kinds handled above.
  return 0;
}

size_t align_up(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

} // namespace

ArgumentBuffer::ArgumentBuffer(Device& d, std::vector<Slot> slots)
    : layout_(std::move(slots)) {
  // Assign each slot's byte offset from the packed layout. Each slot
  // aligns to its own size (so 8-byte scalars land on 8-byte
  // boundaries, 16-byte buffer-ptr entries on 16-byte boundaries, etc.).
  size_t offset = 0;
  for (auto& s : layout_) {
    const size_t sz = slot_size(s.kind);
    offset = align_up(offset, sz);
    s.byte_offset = offset;
    offset += sz;
  }
  // Round total size up to 16 B so downstream shaders can assume the
  // buffer is safe to read with 16-byte vector loads.
  size_ = align_up(offset, 16);
  if (size_ == 0) {
    // Metal rejects zero-length buffers; allocate a single 16-byte
    // slot so `mtl_buffer()` is always non-null.
    size_ = 16;
  }

  buffer_ = buffer_pool().acquire(d, size_);
  // Zero the contents so callers that never write to a slot still
  // see defined values on the GPU. (The pool may hand back a buffer
  // whose contents reflect a prior AB's layout; zeroing ensures
  // unwritten slots land at well-defined values.)
  std::memset(buffer_->contents(), 0, size_);
}

ArgumentBuffer::~ArgumentBuffer() {
  if (buffer_) {
    buffer_pool().release(size_, std::move(buffer_));
  }
}

void ArgumentBuffer::set_scalar32(int slot, uint32_t value) {
  if (slot < 0 || static_cast<size_t>(slot) >= layout_.size()) {
    throw std::out_of_range(
        "[metal::ArgumentBuffer::set_scalar32] slot index out of range");
  }
  const auto& s = layout_[slot];
  if (s.kind != Slot::Kind::Scalar32) {
    throw std::invalid_argument(
        "[metal::ArgumentBuffer::set_scalar32] slot type mismatch");
  }
  std::memcpy(contents_() + s.byte_offset, &value, sizeof(value));
}

void ArgumentBuffer::set_scalar64(int slot, uint64_t value) {
  if (slot < 0 || static_cast<size_t>(slot) >= layout_.size()) {
    throw std::out_of_range(
        "[metal::ArgumentBuffer::set_scalar64] slot index out of range");
  }
  const auto& s = layout_[slot];
  if (s.kind != Slot::Kind::Scalar64) {
    throw std::invalid_argument(
        "[metal::ArgumentBuffer::set_scalar64] slot type mismatch");
  }
  std::memcpy(contents_() + s.byte_offset, &value, sizeof(value));
}

void ArgumentBuffer::set_float32(int slot, float value) {
  if (slot < 0 || static_cast<size_t>(slot) >= layout_.size()) {
    throw std::out_of_range(
        "[metal::ArgumentBuffer::set_float32] slot index out of range");
  }
  const auto& s = layout_[slot];
  if (s.kind != Slot::Kind::Float32) {
    throw std::invalid_argument(
        "[metal::ArgumentBuffer::set_float32] slot type mismatch");
  }
  std::memcpy(contents_() + s.byte_offset, &value, sizeof(value));
}

void ArgumentBuffer::set_buffer_ptr(
    int slot,
    const MTL::Buffer* buf,
    int64_t offset) {
  if (slot < 0 || static_cast<size_t>(slot) >= layout_.size()) {
    throw std::out_of_range(
        "[metal::ArgumentBuffer::set_buffer_ptr] slot index out of range");
  }
  const auto& s = layout_[slot];
  if (s.kind != Slot::Kind::BufferPtrOffset) {
    throw std::invalid_argument(
        "[metal::ArgumentBuffer::set_buffer_ptr] slot type mismatch");
  }
  if (!buf) {
    throw std::invalid_argument(
        "[metal::ArgumentBuffer::set_buffer_ptr] null buffer");
  }
  // Metal API: `gpuAddress()` returns the GPU-visible virtual address
  // of the buffer. Requires macOS 13+ / iOS 16+ which is a floor
  // below our minimum deployment target.
  BufferPtrOffsetEntry e{
      buf->gpuAddress(),
      static_cast<uint64_t>(offset),
  };
  std::memcpy(contents_() + s.byte_offset, &e, sizeof(e));
}

} // namespace mlx::core::metal
