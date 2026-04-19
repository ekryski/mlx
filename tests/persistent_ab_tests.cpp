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
#include <cstdlib>
#include <cstring>

#include "doctest/doctest.h"

#include <Metal/Metal.hpp>

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/persistent_ab.h"
#include "mlx/fast.h"

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

// --------------------------------------------------------------------------
// End-to-end: RMSNorm with a shared PersistentAb matches transient-AB path.
// --------------------------------------------------------------------------
//
// This is the real integration validation for Step 2. We call the
// AB-participating `rms_norm` overload twice on different inputs,
// both times passing the SAME PersistentAb. Output must match what
// the transient-AB path produces for the same inputs. This proves:
//
//   1. Reusing a PersistentAb across calls doesn't corrupt contents.
//   2. eval_gpu correctly rewrites the handle's contents per call.
//   3. The AB-handle dispatch path is numerically equivalent to the
//      fresh-AB path (and by extension, to the non-AB legacy path —
//      the 8-primitive AB stack has been sweep-tested against legacy).
//
// Requires MLX_METAL_AB=1 (or MLX_METAL_ICB=1) because the AB code
// paths are gated behind it. Gracefully skips when the env is off.

namespace {
bool ab_env_on() {
  const char* a = std::getenv("MLX_METAL_AB");
  const char* i = std::getenv("MLX_METAL_ICB");
  return (a && a[0] == '1') || (i && i[0] == '1');
}
} // namespace

TEST_CASE(
    "persistent_ab: RMSNorm with reused handle matches transient-AB output") {
  if (!ab_env_on()) {
    MESSAGE(
        "skipping — AB path gated by MLX_METAL_AB=1 / MLX_METAL_ICB=1");
    return;
  }

  // Skeleton RMSNorm shapes: batch=1, seq=1, hidden=128.
  auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
  const int hidden = 128;
  const float eps = 1e-5f;

  auto key = mlx::core::random::key(0xCAFEBABEull);
  auto keys = mlx::core::random::split(key, 4, stream);

  auto x_a = mlx::core::random::normal(
      {1, 1, hidden}, mlx::core::float32, 0.0f, 1.0f,
      mlx::core::take(keys, 0, 0, stream), stream);
  auto x_b = mlx::core::random::normal(
      {1, 1, hidden}, mlx::core::float32, 0.0f, 1.0f,
      mlx::core::take(keys, 1, 0, stream), stream);
  auto w = mlx::core::random::normal(
      {hidden}, mlx::core::float32, 0.0f, 1.0f,
      mlx::core::take(keys, 2, 0, stream), stream);
  mlx::core::eval(x_a);
  mlx::core::eval(x_b);
  mlx::core::eval(w);

  // Reference: two calls through the regular (transient-AB) overload.
  auto y_a_ref = mlx::core::fast::rms_norm(x_a, w, eps, stream);
  auto y_b_ref = mlx::core::fast::rms_norm(x_b, w, eps, stream);
  mlx::core::eval(y_a_ref);
  mlx::core::eval(y_b_ref);

  // Test: two calls sharing a single PersistentAb handle. The layout
  // must match what RMSNorm::eval_gpu expects.
  auto& d = device(mlx::core::Device::gpu);
  auto handle = std::make_shared<PersistentAb>(
      d,
      std::vector<Slot>{
          {Slot::Kind::BufferPtrOffset, 0, "x"},
          {Slot::Kind::BufferPtrOffset, 0, "w"},
          {Slot::Kind::BufferPtrOffset, 0, "out"},
          {Slot::Kind::Float32, 0, "eps"},
          {Slot::Kind::Scalar32, 0, "axis_size"},
          {Slot::Kind::Scalar32, 0, "w_stride"},
      });
  auto* handle_mtl = handle->mtl_buffer();
  REQUIRE(handle_mtl);

  auto y_a_pers = mlx::core::fast::rms_norm(x_a, w, eps, handle, stream);
  auto y_b_pers = mlx::core::fast::rms_norm(x_b, w, eps, handle, stream);
  mlx::core::eval(y_a_pers);
  mlx::core::eval(y_b_pers);

  // Handle's MTLBuffer pointer must still be the same we observed
  // before the calls — ICB replay depends on this invariant.
  CHECK(handle->mtl_buffer() == handle_mtl);

  // Outputs must match the transient-AB reference bit-for-bit (same
  // kernel, same inputs, same scalars — both paths write the same
  // AB contents before dispatch).
  CHECK(mlx::core::allclose(y_a_pers, y_a_ref, 0.0, 0.0).item<bool>());
  CHECK(mlx::core::allclose(y_b_pers, y_b_ref, 0.0, 0.0).item<bool>());
}

TEST_CASE(
    "persistent_ab: RMSNorm throws on mismatched handle slot count") {
  if (!ab_env_on()) {
    MESSAGE("skipping — AB path gated by MLX_METAL_AB=1 / MLX_METAL_ICB=1");
    return;
  }

  auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& d = device(mlx::core::Device::gpu);

  // Wrong layout — only 3 slots instead of 6.
  auto handle = std::make_shared<PersistentAb>(
      d,
      std::vector<Slot>{
          {Slot::Kind::BufferPtrOffset, 0, "x"},
          {Slot::Kind::BufferPtrOffset, 0, "w"},
          {Slot::Kind::BufferPtrOffset, 0, "out"},
      });

  auto x = mlx::core::ones({1, 1, 128}, mlx::core::float32, stream);
  auto w = mlx::core::ones({128}, mlx::core::float32, stream);
  mlx::core::eval(x);
  mlx::core::eval(w);

  auto y = mlx::core::fast::rms_norm(x, w, 1e-5f, handle, stream);
  CHECK_THROWS_AS(mlx::core::eval(y), std::runtime_error);
}
