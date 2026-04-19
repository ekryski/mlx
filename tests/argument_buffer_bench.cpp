// Copyright © 2026 Apple Inc.
//
// Argument Buffer micro-benchmark: measures the CPU cost of encoding a
// compute dispatch under two binding strategies.
//
//   (A) legacy: 6 individual bindings per dispatch
//       (3 `setBuffer` for in/w/out + 3 `setBytes` for eps/axis/stride),
//       matching today's `RMSNorm::eval_gpu`.
//
//   (B) argument buffer: one `setBuffer` per dispatch, binding a packed
//       `RmsArgs`-shaped buffer at buffer(0). Scalars + pointers live
//       inside the buffer and the kernel reads them at execution time.
//
// Both paths dispatch the same logical workload (1-thread RMS-shaped
// kernels whose only real job is to touch all the bound arguments, so
// the Metal driver can't optimize the bindings away).
//
// Report: µs/dispatch for each path, the ratio, and the extra
// per-dispatch overhead the legacy path carries vs AB. Feeds the
// Phase-1 go/no-go heuristic in
// benchmarks/notes/ab-rmsnorm-pilot-2026-04-17.md (≥ 1.5x drop ⇒
// proceed to Phase 2).

#include <chrono>
#include <cstdint>
#include <sstream>
#include <vector>

#include "doctest/doctest.h"

#include <Metal/Metal.hpp>

#include "mlx/backend/metal/device.h"

using namespace std::chrono;

namespace {

constexpr size_t kDispatches = 1000; // per encoding-cost measurement
constexpr size_t kWarmupIters = 3;
constexpr size_t kMeasuredIters = 20;

// Two tiny kernels with the same compute body but different binding
// strategies. The AB kernel's argument struct mirrors the C++
// `ArgumentBuffer` packing used by RMSNorm's AB path:
//   BufferPtrOffset { u64 addr; u64 offset; } × 3   (x, w, out)
//   float eps; uint axis_size; uint w_stride; uint _pad;
const char* kKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Legacy signature — 6 individual kernel arguments.
kernel void bench_legacy(
    const device float* x   [[buffer(0)]],
    const device float* w   [[buffer(1)]],
    device       float* out [[buffer(2)]],
    constant float& eps        [[buffer(3)]],
    constant uint&  axis_size  [[buffer(4)]],
    constant uint&  w_stride   [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  // Touch every argument so the driver cannot optimize any of them
  // away. The single-thread dispatch keeps the GPU-side cost tiny so
  // the CPU encoding is the dominant term.
  if (tid == 0) {
    out[0] = x[0] * w[w_stride] + eps * float(axis_size);
  }
}

struct BufferPtrOffset {
  uint64_t addr;
  uint64_t offset;
};
struct RmsArgs {
  BufferPtrOffset x;
  BufferPtrOffset w;
  BufferPtrOffset out;
  float eps;
  uint axis_size;
  uint w_stride;
  uint _pad;
};

// AB signature — one packed argument struct at buffer(0).
kernel void bench_ab(
    constant const RmsArgs& args [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
  const device float* x =
      reinterpret_cast<const device float*>(args.x.addr + args.x.offset);
  const device float* w =
      reinterpret_cast<const device float*>(args.w.addr + args.w.offset);
  device float* out =
      reinterpret_cast<device float*>(args.out.addr + args.out.offset);
  if (tid == 0) {
    out[0] = x[0] * w[args.w_stride] + args.eps * float(args.axis_size);
  }
}
)MSL";

struct Bench {
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::Library> library;
  NS::SharedPtr<MTL::ComputePipelineState> pso_legacy;
  NS::SharedPtr<MTL::ComputePipelineState> pso_ab;
  NS::SharedPtr<MTL::CommandQueue> queue;
  NS::SharedPtr<MTL::Buffer> x_buf;
  NS::SharedPtr<MTL::Buffer> w_buf;
  NS::SharedPtr<MTL::Buffer> out_buf;
  NS::SharedPtr<MTL::Buffer> ab_buf;

  // Legacy scalars (CPU copies, passed via setBytes each dispatch).
  float eps = 1e-6f;
  uint32_t axis_size = 1024;
  uint32_t w_stride = 1;
};

NS::SharedPtr<MTL::ComputePipelineState>
compile_pso(MTL::Device* dev, MTL::Library* lib, const char* name) {
  auto fn_name = NS::String::string(name, NS::UTF8StringEncoding);
  auto fn = NS::TransferPtr(lib->newFunction(fn_name));
  REQUIRE(fn);
  NS::Error* err = nullptr;
  auto pso = NS::TransferPtr(
      dev->newComputePipelineState(fn.get(), &err));
  REQUIRE_MESSAGE(
      pso,
      "pipeline creation failed: ",
      (err ? err->localizedDescription()->utf8String() : "null"));
  return pso;
}

Bench make_bench() {
  Bench b;
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  b.device = NS::RetainPtr(d.mtl_device());
  REQUIRE(b.device);

  auto src = NS::String::string(kKernelSource, NS::UTF8StringEncoding);
  NS::Error* err = nullptr;
  b.library = NS::TransferPtr(b.device->newLibrary(src, nullptr, &err));
  REQUIRE_MESSAGE(
      b.library,
      "library compile failed: ",
      (err ? err->localizedDescription()->utf8String() : "null"));

  b.pso_legacy = compile_pso(b.device.get(), b.library.get(), "bench_legacy");
  b.pso_ab = compile_pso(b.device.get(), b.library.get(), "bench_ab");

  b.queue = NS::TransferPtr(b.device->newCommandQueue());
  REQUIRE(b.queue);

  const size_t nbytes = 1024 * sizeof(float);
  b.x_buf = NS::TransferPtr(
      b.device->newBuffer(nbytes, MTL::ResourceStorageModeShared));
  b.w_buf = NS::TransferPtr(
      b.device->newBuffer(nbytes, MTL::ResourceStorageModeShared));
  b.out_buf = NS::TransferPtr(
      b.device->newBuffer(nbytes, MTL::ResourceStorageModeShared));
  REQUIRE(b.x_buf);
  REQUIRE(b.w_buf);
  REQUIRE(b.out_buf);

  // Pack the argument struct for the AB path. 16 × 3 + 4 × 4 = 64 B.
  struct BufferPtrOffset {
    uint64_t addr;
    uint64_t offset;
  };
  struct RmsArgs {
    BufferPtrOffset x;
    BufferPtrOffset w;
    BufferPtrOffset out;
    float eps;
    uint32_t axis_size;
    uint32_t w_stride;
    uint32_t _pad;
  };
  b.ab_buf = NS::TransferPtr(
      b.device->newBuffer(sizeof(RmsArgs), MTL::ResourceStorageModeShared));
  REQUIRE(b.ab_buf);
  auto* args = static_cast<RmsArgs*>(b.ab_buf->contents());
  args->x = {b.x_buf->gpuAddress(), 0};
  args->w = {b.w_buf->gpuAddress(), 0};
  args->out = {b.out_buf->gpuAddress(), 0};
  args->eps = b.eps;
  args->axis_size = b.axis_size;
  args->w_stride = b.w_stride;
  args->_pad = 0;

  return b;
}

double run_legacy_once(Bench& b) {
  auto t0 = steady_clock::now();
  auto* cbuf = b.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  for (size_t i = 0; i < kDispatches; ++i) {
    enc->setComputePipelineState(b.pso_legacy.get());
    enc->setBuffer(b.x_buf.get(), 0, 0);
    enc->setBuffer(b.w_buf.get(), 0, 1);
    enc->setBuffer(b.out_buf.get(), 0, 2);
    enc->setBytes(&b.eps, sizeof(b.eps), 3);
    enc->setBytes(&b.axis_size, sizeof(b.axis_size), 4);
    enc->setBytes(&b.w_stride, sizeof(b.w_stride), 5);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  }
  enc->endEncoding();
  auto t1 = steady_clock::now();
  cbuf->commit();
  cbuf->waitUntilCompleted();
  return duration<double, std::micro>(t1 - t0).count();
}

double run_ab_once(Bench& b) {
  auto t0 = steady_clock::now();
  auto* cbuf = b.queue->commandBuffer();
  auto* enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  // useResource on the backing buffers — the AB holds raw GPU addresses
  // so Metal needs an explicit residency declaration to keep them
  // mapped.
  enc->useResource(b.x_buf.get(), MTL::ResourceUsageRead);
  enc->useResource(b.w_buf.get(), MTL::ResourceUsageRead);
  enc->useResource(b.out_buf.get(), MTL::ResourceUsageWrite);
  for (size_t i = 0; i < kDispatches; ++i) {
    enc->setComputePipelineState(b.pso_ab.get());
    enc->setBuffer(b.ab_buf.get(), 0, 0);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  }
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

} // namespace

TEST_CASE("argument buffer bench: legacy vs AB per-dispatch encoding cost") {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  Bench b = make_bench();

  for (size_t i = 0; i < kWarmupIters; ++i) {
    run_legacy_once(b);
    run_ab_once(b);
  }

  std::vector<double> legacy_times;
  legacy_times.reserve(kMeasuredIters);
  for (size_t i = 0; i < kMeasuredIters; ++i) {
    legacy_times.push_back(run_legacy_once(b));
  }

  std::vector<double> ab_times;
  ab_times.reserve(kMeasuredIters);
  for (size_t i = 0; i < kMeasuredIters; ++i) {
    ab_times.push_back(run_ab_once(b));
  }

  double legacy_med = median(legacy_times);
  double ab_med = median(ab_times);
  double legacy_per = legacy_med / kDispatches;
  double ab_per = ab_med / kDispatches;
  double ratio = legacy_med / ab_med;
  double saved_per = legacy_per - ab_per;

  std::ostringstream summary;
  summary << "\n"
          << "  Argument Buffer microbench (dispatches/iter: " << kDispatches
          << ")\n"
          << "  -----------------------------------------------------------\n"
          << "  Legacy (6 bindings)  median total: " << legacy_med << " us "
          << "(" << legacy_per << " us/dispatch)\n"
          << "  AB (1 binding)       median total: " << ab_med << " us "
          << "(" << ab_per << " us/dispatch)\n"
          << "  Ratio (legacy/ab):                " << ratio << "x\n"
          << "  Per-dispatch saved:               " << saved_per << " us\n";
  MESSAGE(summary.str());

  // Soft sanity check: AB should not be slower than legacy. The pilot
  // plan's go/no-go threshold (≥ 1.5x) is evaluated by the benchmark
  // note, not asserted here.
  CHECK(ab_med <= legacy_med * 1.25);
}
