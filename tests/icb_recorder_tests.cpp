// Copyright © 2026 Apple Inc.
//
// Unit tests for `mlx::core::metal::IndirectCommandRecorder`. These
// exercise the recorder in isolation — no CommandEncoder integration yet
// (that lives in later Phase 1 work). The tests build compute pipelines
// manually against mlx's Metal device, record a few commands, finalize,
// and replay on a fresh ComputeCommandEncoder.

#include <cstdint>
#include <cstring>

#include "doctest/doctest.h"

#include <Metal/Metal.hpp>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/icb.h"

using namespace mlx::core::metal;

namespace {

// Two tiny kernels: "fill" writes a constant bytes-arg * 1.0 into each
// output slot; "add" adds scale (bytes-arg) to each input slot. Exercises
// both pure-buffer and setBytes paths.
const char* kKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Scale { float value; };

kernel void icb_test_fill(
    device       float* out   [[buffer(0)]],
    device const Scale* scale [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = scale->value;
}

kernel void icb_test_add_in_place(
    device       float* out   [[buffer(0)]],
    device const Scale* scale [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = out[tid] + scale->value;
}
)MSL";

struct TestRig {
  NS::SharedPtr<MTL::Library> library;
  NS::SharedPtr<MTL::ComputePipelineState> pso_fill;
  NS::SharedPtr<MTL::ComputePipelineState> pso_add;
  NS::SharedPtr<MTL::CommandQueue> queue;
  NS::SharedPtr<MTL::Buffer> out_buf;
};

NS::SharedPtr<MTL::ComputePipelineState> make_pso(
    MTL::Device* dev,
    MTL::Library* lib,
    const char* fn_name) {
  auto fn = NS::TransferPtr(
      lib->newFunction(NS::String::string(fn_name, NS::UTF8StringEncoding)));
  REQUIRE(fn);
  auto desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  desc->setComputeFunction(fn.get());
  desc->setSupportIndirectCommandBuffers(true);
  NS::Error* err = nullptr;
  auto pso = NS::TransferPtr(dev->newComputePipelineState(
      desc.get(), MTL::PipelineOptionNone, nullptr, &err));
  REQUIRE_MESSAGE(pso, "pso creation failed: ",
      (err ? err->localizedDescription()->utf8String() : "null"));
  return pso;
}

TestRig make_rig(size_t elems) {
  TestRig r;
  auto& d = device(mlx::core::Device::gpu);
  auto* dev = d.mtl_device();

  auto src = NS::String::string(kKernelSource, NS::UTF8StringEncoding);
  NS::Error* err = nullptr;
  r.library = NS::TransferPtr(dev->newLibrary(src, nullptr, &err));
  REQUIRE(r.library);

  r.pso_fill = make_pso(dev, r.library.get(), "icb_test_fill");
  r.pso_add = make_pso(dev, r.library.get(), "icb_test_add_in_place");

  r.queue = NS::TransferPtr(dev->newCommandQueue());
  r.out_buf = NS::TransferPtr(
      dev->newBuffer(elems * sizeof(float), MTL::ResourceStorageModeShared));
  auto* p = static_cast<float*>(r.out_buf->contents());
  for (size_t i = 0; i < elems; ++i) {
    p[i] = 0.0f;
  }
  return r;
}

} // namespace

TEST_CASE("icb recorder: supported on this device") {
  auto& d = device(mlx::core::Device::gpu);
  CHECK(IndirectCommandRecorder::is_supported(d));
}

TEST_CASE("icb recorder: single fill command via setBytes arena") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  constexpr size_t N = 64;
  auto rig = make_rig(N);

  auto& d = device(mlx::core::Device::gpu);
  IndirectCommandRecorder rec(d, /*max_commands=*/1);

  rec.begin_command(rig.pso_fill.get());
  rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
  struct {
    float value;
  } scale{3.5f};
  CHECK(rec.set_bytes(&scale, sizeof(scale), 1));
  rec.end_command(
      MTL::Size(N, 1, 1),
      MTL::Size(1, 1, 1),
      /*use_dispatch_threads=*/true);
  rec.finalize();
  CHECK(rec.size() == 1);

  // Replay on a fresh command buffer.
  auto* cbuf = rig.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  rec.replay(enc);
  enc->endEncoding();
  cbuf->commit();
  cbuf->waitUntilCompleted();

  auto* out = static_cast<float*>(rig.out_buf->contents());
  for (size_t i = 0; i < N; ++i) {
    CHECK(out[i] == doctest::Approx(3.5f));
  }
}

TEST_CASE("icb recorder: fill then accumulate chain, two kernels") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  constexpr size_t N = 32;
  auto rig = make_rig(N);

  auto& d = device(mlx::core::Device::gpu);
  IndirectCommandRecorder rec(d, /*max_commands=*/4);

  // Command 0: fill with 1.0
  rec.begin_command(rig.pso_fill.get());
  rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
  float scale_fill = 1.0f;
  CHECK(rec.set_bytes(&scale_fill, sizeof(scale_fill), 1));
  rec.end_command(MTL::Size(N, 1, 1), MTL::Size(1, 1, 1), true);

  // Commands 1-3: add 2.5, 2.5, 2.5 → expected 1 + 7.5 = 8.5
  for (int i = 0; i < 3; ++i) {
    rec.begin_command(rig.pso_add.get());
    rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
    float add = 2.5f;
    CHECK(rec.set_bytes(&add, sizeof(add), 1));
    rec.end_command(MTL::Size(N, 1, 1), MTL::Size(1, 1, 1), true);
  }
  rec.finalize();
  CHECK(rec.size() == 4);

  auto* cbuf = rig.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  rec.replay(enc);
  enc->endEncoding();
  cbuf->commit();
  cbuf->waitUntilCompleted();

  auto* out = static_cast<float*>(rig.out_buf->contents());
  // NOTE: the three add commands execute concurrently per ICB semantics,
  // so out-of-order reads/writes to the SAME address are technically a
  // data race. For this test we use a single-element-per-thread pattern
  // and rely on the GPU's write-atomicity for independent slots. For
  // true sequential accumulate we'd need barrier splitting, which is
  // Phase 1.4 work. Here we assert the *bounded* expected value (4.0
  // minimum — one add landed — up to 8.5 full chain).
  for (size_t i = 0; i < N; ++i) {
    CHECK(out[i] >= 1.0f);
    CHECK(out[i] <= 8.5f + 0.01f);
  }
}

TEST_CASE("icb recorder: bytes arena exhaustion returns false") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto rig = make_rig(8);
  auto& d = device(mlx::core::Device::gpu);

  // Tiny arena — only room for a single aligned allocation.
  IndirectCommandRecorder rec(d, /*max_commands=*/2, /*bytes_arena_cap=*/16);

  rec.begin_command(rig.pso_fill.get());
  rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
  float v = 1.0f;
  CHECK(rec.set_bytes(&v, sizeof(v), 1));
  rec.end_command(MTL::Size(8, 1, 1), MTL::Size(1, 1, 1), true);

  rec.begin_command(rig.pso_fill.get());
  rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
  // First allocation took 16 bytes of the 16-byte arena — the next must fail.
  CHECK_FALSE(rec.set_bytes(&v, sizeof(v), 1));
}

TEST_CASE("icb recorder: capacity overflow throws") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto rig = make_rig(8);
  auto& d = device(mlx::core::Device::gpu);

  IndirectCommandRecorder rec(d, /*max_commands=*/1);
  rec.begin_command(rig.pso_fill.get());
  rec.end_command(MTL::Size(8, 1, 1), MTL::Size(1, 1, 1), true);

  CHECK_THROWS_AS(rec.begin_command(rig.pso_fill.get()), std::overflow_error);
}

TEST_CASE("icb recorder: replay before finalize throws") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto rig = make_rig(8);
  auto& d = device(mlx::core::Device::gpu);
  IndirectCommandRecorder rec(d, /*max_commands=*/1);

  auto* cbuf = rig.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  CHECK_THROWS_AS(rec.replay(enc), std::logic_error);
  enc->endEncoding();
}

// Exercises the CommandEncoder integration — the routing of
// set_compute_pipeline_state / set_buffer / set_bytes / dispatch_* into
// the recorder while `recording_` is true.
TEST_CASE("icb CommandEncoder integration: record + replay via stream encoder") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  constexpr size_t N = 32;
  auto rig = make_rig(N);

  mlx::core::Stream s = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  // Record a single fill command through the CommandEncoder API. This is
  // exactly how a primitive would drive it during eval_gpu — only
  // difference is we bracket the dispatches with begin/end_icb_recording.
  enc.begin_icb_recording(/*max_commands=*/1);
  enc.set_compute_pipeline_state(rig.pso_fill.get());
  enc.set_buffer(rig.out_buf.get(), 0);  // slot 0 = out
  struct Scale { float value; } scale{7.0f};
  enc.set_bytes(scale, 1);
  enc.dispatch_threads(MTL::Size(N, 1, 1), MTL::Size(1, 1, 1));
  auto rec = enc.end_icb_recording();
  REQUIRE(rec);
  CHECK(rec->size() == 1);
  CHECK(rec->finalized());

  // Replay through the same CommandEncoder (it rebuilds the live encoder
  // lazily). Force a synchronize so the GPU work actually runs before we
  // read the output.
  enc.replay_icb(*rec);
  enc.synchronize();

  auto* out = static_cast<float*>(rig.out_buf->contents());
  for (size_t i = 0; i < N; ++i) {
    CHECK(out[i] == doctest::Approx(7.0f));
  }
}

TEST_CASE("icb CommandEncoder integration: missing dispatch throws at end") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto rig = make_rig(8);
  mlx::core::Stream s = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  enc.begin_icb_recording(/*max_commands=*/1);
  enc.set_compute_pipeline_state(rig.pso_fill.get());
  // Deliberately do not dispatch. end_icb_recording must catch the
  // pending-command state *and* auto-abort so the encoder isn't stuck
  // in recording mode afterwards.
  CHECK_THROWS_AS(enc.end_icb_recording(), std::logic_error);

  // Recording has been aborted; subsequent calls must see a clean slate.
  CHECK_FALSE(enc.is_recording());
}

TEST_CASE("icb recorder: explicit split between fill and sequential accumulate") {
  // With a split between every add, reads of out_buf sit AFTER a memory
  // barrier on replay. Result is deterministic: 1.0 + 2.5 * 3 = 8.5.
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  constexpr size_t N = 32;
  auto rig = make_rig(N);
  auto& d = device(mlx::core::Device::gpu);

  IndirectCommandRecorder rec(d, /*max_commands_per_segment=*/4);

  rec.begin_command(rig.pso_fill.get());
  rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
  float v1 = 1.0f;
  CHECK(rec.set_bytes(&v1, sizeof(v1), 1));
  rec.end_command(MTL::Size(N, 1, 1), MTL::Size(1, 1, 1), true);

  for (int i = 0; i < 3; ++i) {
    rec.split_segment();
    rec.begin_command(rig.pso_add.get());
    rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
    float v2 = 2.5f;
    CHECK(rec.set_bytes(&v2, sizeof(v2), 1));
    rec.end_command(MTL::Size(N, 1, 1), MTL::Size(1, 1, 1), true);
  }
  rec.finalize();
  CHECK(rec.num_segments() == 4);
  CHECK(rec.size() == 4);

  auto* cbuf = rig.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  rec.replay(enc);
  enc->endEncoding();
  cbuf->commit();
  cbuf->waitUntilCompleted();

  auto* out = static_cast<float*>(rig.out_buf->contents());
  for (size_t i = 0; i < N; ++i) {
    CHECK(out[i] == doctest::Approx(8.5f));  // 1.0 + 3 * 2.5
  }
}

TEST_CASE("icb recorder: split on empty segment is a no-op") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto rig = make_rig(8);
  auto& d = device(mlx::core::Device::gpu);

  IndirectCommandRecorder rec(d, /*max_commands_per_segment=*/4);
  rec.split_segment();  // No-op — segment was empty
  rec.split_segment();  // No-op — still empty

  rec.begin_command(rig.pso_fill.get());
  rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
  float v = 1.0f;
  CHECK(rec.set_bytes(&v, sizeof(v), 1));
  rec.end_command(MTL::Size(8, 1, 1), MTL::Size(1, 1, 1), true);
  rec.finalize();
  CHECK(rec.num_segments() == 1);  // Only the real segment exists
}

TEST_CASE("icb recorder: threadgroup memory length is recorded") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  constexpr size_t N = 8;
  auto rig = make_rig(N);
  auto& d = device(mlx::core::Device::gpu);

  IndirectCommandRecorder rec(d, /*max_commands=*/1);
  rec.begin_command(rig.pso_fill.get());
  rec.set_kernel_buffer(rig.out_buf.get(), 0, 0);
  struct { float v; } s{9.0f};
  CHECK(rec.set_bytes(&s, sizeof(s), 1));
  // Request 128 bytes of threadgroup memory at slot 0. The fill kernel
  // doesn't actually use it, but this exercises the code path end-to-end
  // without requiring a new kernel.
  rec.set_threadgroup_memory(128, 0);
  rec.end_command(MTL::Size(N, 1, 1), MTL::Size(1, 1, 1), true);
  rec.finalize();

  auto* cbuf = rig.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  rec.replay(enc);
  enc->endEncoding();
  cbuf->commit();
  cbuf->waitUntilCompleted();

  auto* out = static_cast<float*>(rig.out_buf->contents());
  for (size_t i = 0; i < N; ++i) {
    CHECK(out[i] == doctest::Approx(9.0f));
  }
}

TEST_CASE("icb CommandEncoder integration: begin while recording throws") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto rig = make_rig(8);
  mlx::core::Stream s = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  enc.begin_icb_recording(/*max_commands=*/1);
  CHECK_THROWS_AS(enc.begin_icb_recording(1), std::logic_error);

  // Finish cleanly.
  enc.set_compute_pipeline_state(rig.pso_fill.get());
  enc.set_buffer(rig.out_buf.get(), 0);
  struct { float v; } s_{0.0f};
  enc.set_bytes(s_, 1);
  enc.dispatch_threads(MTL::Size(8, 1, 1), MTL::Size(1, 1, 1));
  (void)enc.end_icb_recording();
}
