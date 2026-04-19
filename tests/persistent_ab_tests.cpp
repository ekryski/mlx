// Copyright © 2026 Apple Inc.
//
// Unit tests for `mlx::core::metal::PersistentAb` — a caller-owned
// ArgumentBuffer variant whose lifetime outlives individual command
// buffers. Used as the building block for Option A (persistent-AB +
// ICB decode-loop replay).
//
// Coverage: construction, slot writes at correct offsets, MTLBuffer
// address stability across re-writes, round-trip read of contents via
// the shared-storage pointer, and the non-copyable / non-movable
// contract.

#include <cstdint>
#include <cstring>

#include "doctest/doctest.h"

#include <Metal/Metal.hpp>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/persistent_ab.h"

using namespace mlx::core::metal;
using Slot = ArgumentBuffer::Slot;

TEST_CASE("persistent_ab: construction allocates a stable MTLBuffer") {
  auto& d = device(mlx::core::Device::gpu);
  PersistentAb ab(
      d,
      {
          {Slot::Kind::Scalar32, 0, "a"},
          {Slot::Kind::Float32, 0, "b"},
          {Slot::Kind::BufferPtrOffset, 0, "c"},
      });
  REQUIRE(ab.mtl_buffer());
  CHECK(ab.size_bytes() >= 16);
  CHECK(ab.layout().size() == 3);
}

TEST_CASE("persistent_ab: mtl_buffer pointer is stable across slot writes") {
  // This is the core contract for ICB replay: the caller records
  // `setBuffer(ab.mtl_buffer(), 0)` once, then updates contents on
  // subsequent steps. The MTLBuffer pointer must not change across
  // setter calls.
  auto& d = device(mlx::core::Device::gpu);
  PersistentAb ab(
      d,
      {
          {Slot::Kind::Scalar32, 0, "x"},
          {Slot::Kind::Scalar32, 0, "y"},
      });

  auto* initial = ab.mtl_buffer();
  REQUIRE(initial);

  ab.set_scalar32(0, 42);
  CHECK(ab.mtl_buffer() == initial);

  ab.set_scalar32(1, 99);
  CHECK(ab.mtl_buffer() == initial);

  ab.set_scalar32(0, 123);
  CHECK(ab.mtl_buffer() == initial);
}

TEST_CASE("persistent_ab: slot writes land at the declared offsets") {
  auto& d = device(mlx::core::Device::gpu);
  PersistentAb ab(
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

TEST_CASE("persistent_ab: buffer_ptr writes gpuAddress + offset") {
  auto& d = device(mlx::core::Device::gpu);
  auto* dev = d.mtl_device();

  auto target = NS::TransferPtr(
      dev->newBuffer(1024, MTL::ResourceStorageModeShared));
  REQUIRE(target);

  PersistentAb ab(d, {{Slot::Kind::BufferPtrOffset, 0, "x"}});
  ab.set_buffer_ptr(0, target.get(), /*offset=*/256);

  const auto* p = static_cast<const uint8_t*>(ab.mtl_buffer()->contents());
  struct Entry {
    uint64_t addr;
    uint64_t offset;
  } e{};
  std::memcpy(&e, p + ab.layout()[0].byte_offset, sizeof(e));

  CHECK(e.addr == target->gpuAddress());
  CHECK(e.offset == 256u);
}

TEST_CASE("persistent_ab: overwriting a slot updates contents in place") {
  // Simulates the per-decode-step update pattern: the same slot gets
  // rewritten on every step with a fresh value. Contents must reflect
  // the latest write; buffer pointer must not have moved.
  auto& d = device(mlx::core::Device::gpu);
  PersistentAb ab(d, {{Slot::Kind::Scalar32, 0, "t_k"}});
  auto* buf = ab.mtl_buffer();
  REQUIRE(buf);

  const auto* p = static_cast<const uint8_t*>(buf->contents());
  const size_t offset = ab.layout()[0].byte_offset;

  for (uint32_t step : {1u, 2u, 3u, 1024u, 1025u, 4096u}) {
    ab.set_scalar32(0, step);
    CHECK(ab.mtl_buffer() == buf);  // buffer pointer stable
    uint32_t read_back;
    std::memcpy(&read_back, p + offset, sizeof(read_back));
    CHECK(read_back == step);
  }
}

TEST_CASE("persistent_ab: type-mismatch setters throw") {
  auto& d = device(mlx::core::Device::gpu);
  PersistentAb ab(
      d,
      {
          {Slot::Kind::Scalar32, 0, "a"},
          {Slot::Kind::Float32, 0, "b"},
      });
  CHECK_THROWS_AS(ab.set_float32(0, 1.0f), std::invalid_argument);
  CHECK_THROWS_AS(ab.set_scalar32(1, 1u), std::invalid_argument);
}

TEST_CASE("persistent_ab: out-of-range slot index throws") {
  auto& d = device(mlx::core::Device::gpu);
  PersistentAb ab(d, {{Slot::Kind::Scalar32, 0, "a"}});
  CHECK_THROWS_AS(ab.set_scalar32(1, 1u), std::out_of_range);
  CHECK_THROWS_AS(ab.set_scalar32(-1, 1u), std::out_of_range);
}

TEST_CASE("persistent_ab: non-copyable and non-movable at compile time") {
  // These static_asserts are the ownership contract: exactly one
  // owner, one MTLBuffer. Hold via shared_ptr / unique_ptr where
  // handle semantics are needed.
  static_assert(
      !std::is_copy_constructible_v<PersistentAb>,
      "PersistentAb must not be copy-constructible");
  static_assert(
      !std::is_copy_assignable_v<PersistentAb>,
      "PersistentAb must not be copy-assignable");
  static_assert(
      !std::is_move_constructible_v<PersistentAb>,
      "PersistentAb must not be move-constructible");
  static_assert(
      !std::is_move_assignable_v<PersistentAb>,
      "PersistentAb must not be move-assignable");
  // Trivial runtime check so the test case counts in the tally.
  CHECK(true);
}

TEST_CASE("persistent_ab: multiple independent instances have distinct buffers") {
  auto& d = device(mlx::core::Device::gpu);
  PersistentAb ab_a(d, {{Slot::Kind::Scalar32, 0, "x"}});
  PersistentAb ab_b(d, {{Slot::Kind::Scalar32, 0, "x"}});
  REQUIRE(ab_a.mtl_buffer());
  REQUIRE(ab_b.mtl_buffer());
  CHECK(ab_a.mtl_buffer() != ab_b.mtl_buffer());
}
