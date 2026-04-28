// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include <atomic>
#include <thread>
#include <vector>

#include "mlx/mlx.h"

using namespace mlx::core;

// Smoke test: concurrent eval on built-in primitives must not crash.
//
// Companion to the lifetime fix in CommandEncoder::set_buffer /
// set_input_array (allocator buffers use
// MTLResourceHazardTrackingModeUntracked and command buffers use
// commandBufferWithUnretainedReferences(); both Apple APIs require the
// application to keep bound buffers alive until the CB completes).
//
// This test does NOT deterministically reproduce the underlying bug —
// built-in primitives like matmul have well-rooted shared_ptr chains
// from the standard ops layer, so the buffer being released mid-flight
// is rare on this code path. The bug surfaces reliably only with
// custom Metal kernels (mlx::fast::metal_kernel) under high concurrency.
// A deterministic regression test using metal_kernel is left as a
// follow-up; this smoke test just guards against obvious regressions
// in the eval path under concurrent threads.
TEST_CASE("test concurrent eval smoke") {
  if (!gpu::is_available()) {
    return;
  }

  constexpr int kThreads = 16;
  constexpr int kIters = 8;

  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&]() {
      try {
        for (int i = 0; i < kIters; ++i) {
          auto x = random::normal({256, 256});
          auto w = random::normal({256, 256});
          auto y = matmul(x, w);
          eval(y);
        }
      } catch (...) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  CHECK_EQ(failures.load(), 0);
}
