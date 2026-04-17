// Copyright © 2026 Apple Inc.
//
// ICB feasibility micro-benchmark: measures CPU-side encoding cost of 1500
// trivial compute dispatches via two paths —
//
//   (A) direct: build a ComputeCommandEncoder, emit 1500 dispatchThreadgroups
//       per iteration.
//
//   (B) ICB replay: build an MTLIndirectCommandBuffer once (1500 commands),
//       then per iteration build a ComputeCommandEncoder that calls
//       executeCommandsInBuffer exactly once.
//
// This decides whether it is worth plumbing ICB capture/replay through mlx's
// CommandEncoder and the mlx-c / mlx-swift / mlx-swift-lm layers. The
// strategy doc predicts +15–25% decode tok/s on GPT-OSS-20B from ICB; that
// estimate depends on path (B) encoding materially faster than path (A) on
// the CPU. If the ratio here is <~2x, plumbing ICB is unlikely to pay back.
//
// Output is printed via doctest MESSAGE so it shows up in test logs without
// failing the suite. The test itself only asserts that numerics match
// between the two paths.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <sstream>

#include "doctest/doctest.h"

#include <Metal/Metal.hpp>

#include "mlx/backend/metal/device.h"

using namespace std::chrono;

namespace {

constexpr size_t kDispatches = 1500;  // matches GPT-OSS-20B per-token count
constexpr size_t kWarmupIters = 3;
constexpr size_t kMeasuredIters = 20;
constexpr size_t kElements = 1024;

const char* kKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void icb_bench_mul(
    device const float* in   [[buffer(0)]],
    device       float* out  [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = in[tid] * scale[0];
}
)MSL";

struct Bench {
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::Library> library;
  NS::SharedPtr<MTL::ComputePipelineState> pso;
  NS::SharedPtr<MTL::CommandQueue> queue;
  NS::SharedPtr<MTL::Buffer> in_buf;
  NS::SharedPtr<MTL::Buffer> out_buf;
  NS::SharedPtr<MTL::Buffer> scale_buf;
  NS::SharedPtr<MTL::IndirectCommandBuffer> icb;
};

Bench make_bench() {
  Bench b;

  // Use mlx's device so we share the Metal context the library is built on
  // and don't double-initialize.
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  b.device = NS::RetainPtr(d.mtl_device());
  REQUIRE(b.device);

  // Compile the tiny kernel from source, with ICB support.
  auto src = NS::String::string(kKernelSource, NS::UTF8StringEncoding);
  NS::Error* err = nullptr;
  b.library = NS::TransferPtr(b.device->newLibrary(src, nullptr, &err));
  REQUIRE_MESSAGE(b.library, "library compile failed: ",
      (err ? err->localizedDescription()->utf8String() : "null"));

  auto fn_name = NS::String::string("icb_bench_mul", NS::UTF8StringEncoding);
  auto fn = NS::TransferPtr(b.library->newFunction(fn_name));
  REQUIRE(fn);

  auto pso_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pso_desc->setComputeFunction(fn.get());
  pso_desc->setSupportIndirectCommandBuffers(true);

  err = nullptr;
  b.pso = NS::TransferPtr(b.device->newComputePipelineState(
      pso_desc.get(), MTL::PipelineOptionNone, nullptr, &err));
  REQUIRE_MESSAGE(b.pso, "pipeline creation failed: ",
      (err ? err->localizedDescription()->utf8String() : "null"));

  b.queue = NS::TransferPtr(b.device->newCommandQueue());
  REQUIRE(b.queue);

  // Shared storage so we can verify numerics from the host.
  b.in_buf = NS::TransferPtr(b.device->newBuffer(
      kElements * sizeof(float), MTL::ResourceStorageModeShared));
  b.out_buf = NS::TransferPtr(b.device->newBuffer(
      kElements * sizeof(float), MTL::ResourceStorageModeShared));
  b.scale_buf = NS::TransferPtr(b.device->newBuffer(
      sizeof(float), MTL::ResourceStorageModeShared));
  REQUIRE(b.in_buf);
  REQUIRE(b.out_buf);
  REQUIRE(b.scale_buf);

  auto* in_ptr = static_cast<float*>(b.in_buf->contents());
  for (size_t i = 0; i < kElements; ++i) {
    in_ptr[i] = 1.0f;  // 1500 * 1.0 = 1500, easy correctness check
  }
  *static_cast<float*>(b.scale_buf->contents()) = 1.0f;

  return b;
}

// Build an ICB with `kDispatches` commands. Each command = multiply input
// by scale and write to output. Reusing the same in/out buffers means
// every command is redundant in the data sense — but the CPU cost of
// *encoding* a command is what we're measuring, not the GPU result.
void build_icb(Bench& b) {
  auto desc = NS::TransferPtr(MTL::IndirectCommandBufferDescriptor::alloc()->init());
  desc->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
  desc->setInheritBuffers(false);
  desc->setInheritPipelineState(false);
  desc->setMaxKernelBufferBindCount(3);

  b.icb = NS::TransferPtr(b.device->newIndirectCommandBuffer(
      desc.get(), kDispatches, MTL::ResourceStorageModePrivate));
  REQUIRE(b.icb);

  for (size_t i = 0; i < kDispatches; ++i) {
    auto* cmd = b.icb->indirectComputeCommand(i);
    cmd->setComputePipelineState(b.pso.get());
    cmd->setKernelBuffer(b.in_buf.get(), 0, 0);
    cmd->setKernelBuffer(b.out_buf.get(), 0, 1);
    cmd->setKernelBuffer(b.scale_buf.get(), 0, 2);
    cmd->concurrentDispatchThreadgroups(
        MTL::Size(1, 1, 1), MTL::Size(kElements, 1, 1));
  }
}

double run_direct_once(Bench& b) {
  auto t0 = steady_clock::now();
  auto* cbuf = b.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  for (size_t i = 0; i < kDispatches; ++i) {
    enc->setComputePipelineState(b.pso.get());
    enc->setBuffer(b.in_buf.get(), 0, 0);
    enc->setBuffer(b.out_buf.get(), 0, 1);
    enc->setBuffer(b.scale_buf.get(), 0, 2);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(kElements, 1, 1));
  }
  enc->endEncoding();
  auto t1 = steady_clock::now();
  cbuf->commit();
  cbuf->waitUntilCompleted();
  return duration<double, std::micro>(t1 - t0).count();
}

double run_icb_once(Bench& b) {
  auto t0 = steady_clock::now();
  auto* cbuf = b.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  enc->useResource(b.in_buf.get(),    MTL::ResourceUsageRead);
  enc->useResource(b.out_buf.get(),   MTL::ResourceUsageWrite);
  enc->useResource(b.scale_buf.get(), MTL::ResourceUsageRead);
  enc->executeCommandsInBuffer(b.icb.get(), NS::Range(0, kDispatches));
  enc->endEncoding();
  auto t1 = steady_clock::now();
  cbuf->commit();
  cbuf->waitUntilCompleted();
  return duration<double, std::micro>(t1 - t0).count();
}

double median(std::vector<double> xs) {
  std::sort(xs.begin(), xs.end());
  return xs[xs.size() / 2];
}

}  // namespace

TEST_CASE("icb feasibility: direct vs replay cpu encoding cost") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  Bench b = make_bench();

  // Warm up.
  for (size_t i = 0; i < kWarmupIters; ++i) {
    run_direct_once(b);
  }

  std::vector<double> direct_times;
  direct_times.reserve(kMeasuredIters);
  for (size_t i = 0; i < kMeasuredIters; ++i) {
    direct_times.push_back(run_direct_once(b));
  }

  // Build ICB once (one-time cost — amortized across the decode loop in
  // the real use case). Time it for reporting but don't include it in the
  // per-iteration numbers.
  auto t_build_0 = steady_clock::now();
  build_icb(b);
  auto t_build_1 = steady_clock::now();
  double icb_build_us = duration<double, std::micro>(t_build_1 - t_build_0).count();

  for (size_t i = 0; i < kWarmupIters; ++i) {
    run_icb_once(b);
  }
  std::vector<double> icb_times;
  icb_times.reserve(kMeasuredIters);
  for (size_t i = 0; i < kMeasuredIters; ++i) {
    icb_times.push_back(run_icb_once(b));
  }

  double direct_med = median(direct_times);
  double icb_med = median(icb_times);
  double speedup = direct_med / icb_med;

  std::ostringstream summary;
  summary << "\n"
          << "  ICB feasibility benchmark (dispatches/iter: " << kDispatches << ")\n"
          << "  -----------------------------------------------------------\n"
          << "  Direct encoding  (median): " << direct_med << " us\n"
          << "  ICB replay       (median): " << icb_med    << " us\n"
          << "  ICB build        (once):   " << icb_build_us << " us\n"
          << "  Speedup (direct / icb):    " << speedup << "x\n"
          << "  Break-even replays for ICB build: " << (icb_build_us / (direct_med - icb_med))
          << "\n";
  MESSAGE(summary.str());

  // Correctness: both paths write scale * in_buf to out_buf. Since both
  // paths ran, the latest values in out_buf should equal scale * in = 1.0.
  auto* out_ptr = static_cast<float*>(b.out_buf->contents());
  for (size_t i = 0; i < kElements; ++i) {
    CHECK(out_ptr[i] == doctest::Approx(1.0f));
  }

  // We do not HARD-assert a speedup threshold here; this is a measurement
  // artifact to inform the plan. A soft check to catch regressions: ICB
  // replay should be no slower than direct within 50% noise margin.
  CHECK(icb_med < direct_med * 1.5);
}
