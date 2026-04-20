// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/persistent_ab.h"

#include <cstring>
#include <stdexcept>

#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

PersistentAb::PersistentAb(
    Device& d,
    std::vector<ArgumentBuffer::Slot> layout)
    : ab_(d, std::move(layout)) {}

PersistentAb::~PersistentAb() = default;

void PersistentAb::set_scalar32(int slot, uint32_t value) {
  ab_.set_scalar32(slot, value);
}

void PersistentAb::set_scalar64(int slot, uint64_t value) {
  ab_.set_scalar64(slot, value);
}

void PersistentAb::set_float32(int slot, float value) {
  ab_.set_float32(slot, value);
}

void PersistentAb::set_buffer_ptr(
    int slot,
    const MTL::Buffer* buf,
    int64_t offset) {
  ab_.set_buffer_ptr(slot, buf, offset);
}

void PersistentAb::set_bytes(int slot, const void* data, size_t size) {
  ab_.set_bytes(slot, data, size);
}

MTL::Buffer* PersistentAb::mtl_buffer() const {
  return ab_.mtl_buffer();
}

const std::vector<ArgumentBuffer::Slot>& PersistentAb::layout() const {
  return ab_.layout();
}

size_t PersistentAb::size_bytes() const {
  return ab_.size_bytes();
}

MTL::Buffer* PersistentAb::scalar_buffer(Device& d) {
  if (!scalar_buffer_) {
    auto* dev = d.mtl_device();
    if (!dev) {
      throw std::runtime_error(
          "[metal::PersistentAb::scalar_buffer] Metal device unavailable");
    }
    // Small 4-byte shared-storage buffer. Not pooled — one per
    // handle for the handle's lifetime. Cheap to allocate on the
    // (very rare) first access; no ongoing cost.
    scalar_buffer_ = NS::TransferPtr(
        dev->newBuffer(4, MTL::ResourceStorageModeShared));
    if (!scalar_buffer_) {
      throw std::runtime_error(
          "[metal::PersistentAb::scalar_buffer] failed to allocate "
          "4-byte shared buffer");
    }
    // Zero-init so stale reads before first write are deterministic.
    uint32_t zero = 0;
    std::memcpy(scalar_buffer_->contents(), &zero, sizeof(uint32_t));
  }
  return scalar_buffer_.get();
}

void PersistentAb::set_scalar_u32(Device& d, uint32_t value) {
  auto* buf = scalar_buffer(d);
  std::memcpy(buf->contents(), &value, sizeof(uint32_t));
}

} // namespace mlx::core::metal
