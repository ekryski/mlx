// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/persistent_ab.h"

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

} // namespace mlx::core::metal
