// Copyright © 2026 Apple Inc.
//
// Unit tests for `mlx::core::metal::ArgumentBuffer` — the typed,
// packed argument-block wrapper that replaces many individual
// setBuffer / setBytes calls for ICB-participating primitives.
//
// Tests cover: layout packing (offset assignment), scalar writes,
// buffer-ptr writes (via gpuAddress), slot-type enforcement, bounds
// checking, and zero-init guarantee.

#include <cstdint>
#include <cstring>

#include "doctest/doctest.h"

#include <Metal/Metal.hpp>

#include "mlx/backend/metal/argument_buffer.h"
#include "mlx/backend/metal/device.h"

using namespace mlx::core::metal;
using Slot = ArgumentBuffer::Slot;

TEST_CASE("argument_buffer: empty layout still allocates a valid buffer") {
  auto& d = device(mlx::core::Device::gpu);
  ArgumentBuffer ab(d, {});
  REQUIRE(ab.mtl_buffer());
  // 16-byte minimum so GPU-side 16-byte loads are safe.
  CHECK(ab.size_bytes() >= 16);
}

TEST_CASE("argument_buffer: packed layout assigns natural-size offsets") {
  auto& d = device(mlx::core::Device::gpu);
  // Declared order: u32, u32, u64, f32, ptr — total
  //   4 (u32) + 4 (u32) + 8 (u64) + 4 (f32) + aligned_up to 16 + 16 (ptr) = 40
  ArgumentBuffer ab(
      d,
      {
          {Slot::Kind::Scalar32, 0, "dim0"},
          {Slot::Kind::Scalar32, 0, "dim1"},
          {Slot::Kind::Scalar64, 0, "offset"},
          {Slot::Kind::Float32, 0, "scale"},
          {Slot::Kind::BufferPtrOffset, 0, "input"},
      });
  const auto& layout = ab.layout();
  REQUIRE(layout.size() == 5);
  CHECK(layout[0].byte_offset == 0);   // u32 @ 0
  CHECK(layout[1].byte_offset == 4);   // u32 @ 4
  CHECK(layout[2].byte_offset == 8);   // u64 @ 8 (aligned)
  CHECK(layout[3].byte_offset == 16);  // f32 @ 16
  // ptr is 16-byte aligned — next available is 20, round up to 32.
  CHECK(layout[4].byte_offset == 32);
  // Total size rounded up to 16: 32 + 16 = 48.
  CHECK(ab.size_bytes() == 48);
}

TEST_CASE("argument_buffer: scalar writes land at the declared offsets") {
  auto& d = device(mlx::core::Device::gpu);
  ArgumentBuffer ab(
      d,
      {
          {Slot::Kind::Scalar32, 0, "a"},
          {Slot::Kind::Scalar64, 0, "b"},
          {Slot::Kind::Float32, 0, "c"},
      });

  ab.set_scalar32(0, 0xDEADBEEFu);
  ab.set_scalar64(1, 0x1122334455667788ull);
  ab.set_float32(2, 3.14159f);

  const auto* p = static_cast<const uint8_t*>(ab.mtl_buffer()->contents());
  uint32_t a;
  uint64_t b;
  float c;
  std::memcpy(&a, p + ab.layout()[0].byte_offset, sizeof(a));
  std::memcpy(&b, p + ab.layout()[1].byte_offset, sizeof(b));
  std::memcpy(&c, p + ab.layout()[2].byte_offset, sizeof(c));

  CHECK(a == 0xDEADBEEFu);
  CHECK(b == 0x1122334455667788ull);
  CHECK(c == doctest::Approx(3.14159f));
}

TEST_CASE("argument_buffer: set_buffer_ptr writes gpuAddress + offset") {
  auto& d = device(mlx::core::Device::gpu);
  auto* dev = d.mtl_device();

  // A small throwaway buffer to reference.
  auto target = NS::TransferPtr(
      dev->newBuffer(1024, MTL::ResourceStorageModeShared));
  REQUIRE(target);

  ArgumentBuffer ab(d, {{Slot::Kind::BufferPtrOffset, 0, "x"}});
  ab.set_buffer_ptr(0, target.get(), /*offset=*/256);

  const auto* p = static_cast<const uint8_t*>(ab.mtl_buffer()->contents());
  struct Entry {
    uint64_t addr;
    uint64_t offset;
  } e{};
  std::memcpy(&e, p + ab.layout()[0].byte_offset, sizeof(e));
  CHECK(e.addr == target->gpuAddress());
  CHECK(e.offset == 256);
}

TEST_CASE("argument_buffer: slot-type mismatch throws") {
  auto& d = device(mlx::core::Device::gpu);
  ArgumentBuffer ab(
      d,
      {
          {Slot::Kind::Scalar32, 0, "a"},
          {Slot::Kind::Float32, 0, "b"},
          {Slot::Kind::BufferPtrOffset, 0, "c"},
      });

  // Each setter expects a specific kind.
  CHECK_THROWS_AS(ab.set_scalar64(0, 1), std::invalid_argument);
  CHECK_THROWS_AS(ab.set_scalar32(1, 1), std::invalid_argument);
  CHECK_THROWS_AS(ab.set_float32(2, 1.0f), std::invalid_argument);
}

TEST_CASE("argument_buffer: out-of-range slot throws") {
  auto& d = device(mlx::core::Device::gpu);
  ArgumentBuffer ab(d, {{Slot::Kind::Scalar32, 0, "only"}});
  CHECK_THROWS_AS(ab.set_scalar32(-1, 1), std::out_of_range);
  CHECK_THROWS_AS(ab.set_scalar32(1, 1), std::out_of_range);
  CHECK_THROWS_AS(ab.set_scalar32(100, 1), std::out_of_range);
}

TEST_CASE("argument_buffer: null buffer in set_buffer_ptr throws") {
  auto& d = device(mlx::core::Device::gpu);
  ArgumentBuffer ab(d, {{Slot::Kind::BufferPtrOffset, 0, "x"}});
  CHECK_THROWS_AS(ab.set_buffer_ptr(0, nullptr, 0), std::invalid_argument);
}

TEST_CASE("argument_buffer: freshly-constructed buffer is zero-initialized") {
  auto& d = device(mlx::core::Device::gpu);
  ArgumentBuffer ab(
      d,
      {
          {Slot::Kind::Scalar32, 0, "a"},
          {Slot::Kind::Scalar64, 0, "b"},
          {Slot::Kind::Float32, 0, "c"},
          {Slot::Kind::BufferPtrOffset, 0, "d"},
      });
  const auto* p = static_cast<const uint8_t*>(ab.mtl_buffer()->contents());
  for (size_t i = 0; i < ab.size_bytes(); ++i) {
    CHECK(p[i] == 0);
  }
}

TEST_CASE("argument_buffer: overwriting a slot preserves other slots") {
  auto& d = device(mlx::core::Device::gpu);
  ArgumentBuffer ab(
      d,
      {
          {Slot::Kind::Scalar32, 0, "a"},
          {Slot::Kind::Scalar32, 0, "b"},
          {Slot::Kind::Scalar32, 0, "c"},
      });
  ab.set_scalar32(0, 111);
  ab.set_scalar32(1, 222);
  ab.set_scalar32(2, 333);
  // Overwrite the middle slot only.
  ab.set_scalar32(1, 999);

  const auto* p = static_cast<const uint8_t*>(ab.mtl_buffer()->contents());
  uint32_t a, b, c;
  std::memcpy(&a, p + ab.layout()[0].byte_offset, sizeof(a));
  std::memcpy(&b, p + ab.layout()[1].byte_offset, sizeof(b));
  std::memcpy(&c, p + ab.layout()[2].byte_offset, sizeof(c));
  CHECK(a == 111);
  CHECK(b == 999);
  CHECK(c == 333);
}
