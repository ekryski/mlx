// Copyright © 2026 Apple Inc.
//
// SDPA regression + baseline harness — Phase 0 of Option C (unified
// vector-SDPA rewrite). See plan docs in mlx-swift-lm at
// benchmarks/notes/sdpa-option-c-plan-2026-04-18.md.
//
// This file is a gate, not a benchmark. Its job is to lock down the
// current multi-variant vector-path SDPA's output + rough latency so
// Phase 1 (unified kernel) and Phase 2 (AB-migrated unified kernel)
// have a shared reference to diff against. All test cases run against
// the legacy code paths only; the unified kernel does not exist yet.
//
// Opt-in: tests gate behind env MLX_SDPA_SWEEP=1 so a default test
// run stays fast. Artifact output path is controlled by
// MLX_SDPA_BASELINE_OUT (defaults to ./sdpa-option-c-baseline.md in
// the test binary's cwd).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/random.h"

using namespace mlx::core;

namespace {

// ---------------------------------------------------------------------------
// Opt-in gating
// ---------------------------------------------------------------------------

bool sweep_enabled() {
  const char* e = std::getenv("MLX_SDPA_SWEEP");
  return e != nullptr && e[0] == '1';
}

// True when the harness should run each case on BOTH the unified
// path and the legacy path (via MLX_SDPA_FORCE_LEGACY) and compare.
// Active when AB gate is on — under AB, the default path is unified,
// and we validate equivalence to legacy via the debug override.
bool ab_mode() {
  const char* ab = std::getenv("MLX_METAL_AB");
  const char* icb = std::getenv("MLX_METAL_ICB");
  return (ab && ab[0] == '1') || (icb && icb[0] == '1');
}

std::string baseline_out_path() {
  const char* e = std::getenv("MLX_SDPA_BASELINE_OUT");
  if (e && *e) return std::string(e);
  return "sdpa-option-c-baseline.md";
}

// Tolerance per dtype for unified-vs-legacy allclose. Legacy path is
// reference; unified may differ at the LSB due to kernel-internal
// ordering (single-pass vs 2-pass across partitions).
std::pair<double, double> tolerance_for(Dtype dt) {
  if (dt == float32) return {1e-5, 1e-5};
  return {1e-3, 1e-3};  // fp16 / bf16
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

const char* dtype_str(Dtype t) {
  if (t == float32) return "f32";
  if (t == float16) return "f16";
  if (t == bfloat16) return "bf16";
  return "?";
}

// Timing: reduce noise from one-off JIT + warmup by running once warm
// and returning the median of N small samples in microseconds.
template <typename F>
double time_us_median(F&& f, int samples = 5) {
  // Warm-up invocation — not timed.
  f();
  eval(array(0.0f));  // fence

  std::vector<double> us;
  us.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    us.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() /
        1000.0);
  }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

// Shape-stable compact checksum: sum of |x| cast to float32. Stable
// across dtypes that share the same mathematical values; sensitive to
// any numeric drift. Materializes the array.
double checksum(const array& a) {
  auto abs_f32 = astype(abs(a), float32);
  auto s = sum(abs_f32);
  eval(s);
  return s.item<float>();
}

// Shape helpers for SDPA convention (b, h, seq, d).
Shape qkv_shape(int b, int h, int seq, int d) {
  return Shape{b, h, seq, d};
}

// Random generator — each case uses a deterministic key derived from
// the case index so sweeps are reproducible across test runs.
array rand_like(const Shape& s, Dtype dt, uint64_t seed) {
  auto k = random::key(seed);
  // normal(shape, dtype, loc, scale, key, stream)
  return random::normal(s, dt, 0.0f, 1.0f, k);
}

// ---------------------------------------------------------------------------
// Case spec + sweep generator
// ---------------------------------------------------------------------------

struct CaseSpec {
  int B, H_q, H_k, L_q, L_k, D;
  bool causal;
  std::string mask_mode;   // "", "causal"
  Dtype dtype;
  const char* label;
};

std::string case_key(const CaseSpec& c) {
  std::ostringstream os;
  os << "B" << c.B
     << "_Hq" << c.H_q
     << "_Hk" << c.H_k
     << "_Lq" << c.L_q
     << "_Lk" << c.L_k
     << "_D" << c.D
     << "_" << (c.causal ? "causal" : "nocausal")
     << "_" << dtype_str(c.dtype);
  return os.str();
}

// Stratified sweep — topology-threshold L_k values get exhaustive
// coverage across causal + dtypes; the remaining axes get diagonal
// sampling. Total ~150 cases at default; scaled down further for
// quick sanity runs when MLX_SDPA_QUICK=1.
std::vector<CaseSpec> build_sweep() {
  // Topology-threshold L_k values (from E2 diagnostic).
  const std::vector<int> L_k_threshold = {1023, 1024, 1025, 4095, 4096, 4097};
  // Non-threshold L_k values, diagonal-sampled.
  const std::vector<int> L_k_diagonal = {1, 32, 64, 96, 128, 256, 768, 2048, 8192};

  const std::vector<Dtype> dtypes = {float32, float16, bfloat16};

  std::vector<CaseSpec> out;

  // 1. Threshold cross-product: every L_k_threshold x {causal, noncausal} x dtypes
  //    on a stable (B=1, H_q=4, H_k=4, L_q=1, D=64) skeleton.
  for (int L_k : L_k_threshold) {
    for (bool causal : {false, true}) {
      for (Dtype dt : dtypes) {
        out.push_back(CaseSpec{
            /*B*/ 1, /*H_q*/ 4, /*H_k*/ 4,
            /*L_q*/ 1, /*L_k*/ L_k, /*D*/ 64,
            causal, causal ? "causal" : "",
            dt, "threshold"});
      }
    }
  }

  // 2. Diagonal sample — vary one axis at a time from a stable base.
  for (int L_k : L_k_diagonal) {
    out.push_back(CaseSpec{1, 4, 4, 1, L_k, 64, true, "causal", float16, "diag_Lk"});
  }
  for (int D : {64, 96, 128, 256}) {
    out.push_back(CaseSpec{1, 4, 4, 1, 128, D, true, "causal", float16, "diag_D"});
  }
  for (int H_q : {1, 4, 8, 32}) {
    int H_k = std::min(H_q, 4);
    out.push_back(CaseSpec{1, H_q, H_k, 1, 128, 64, true, "causal", float16, "diag_Hq"});
  }
  for (int H_k_factor : {1, 2, 4, 8}) {
    int H_k = std::max(1, 8 / H_k_factor);
    out.push_back(CaseSpec{1, 8, H_k, 1, 128, 64, true, "causal", float16, "diag_gqa"});
  }
  for (int B : {1, 2, 4}) {
    out.push_back(CaseSpec{B, 4, 4, 1, 128, 64, true, "causal", float16, "diag_B"});
  }
  for (int L_q : {1, 4, 8}) {
    out.push_back(CaseSpec{1, 4, 4, L_q, 128, 64, true, "causal", float16, "diag_Lq"});
  }

  // Optional reduction for quick sanity runs.
  const char* quick = std::getenv("MLX_SDPA_QUICK");
  if (quick && quick[0] == '1') {
    const size_t n = std::min<size_t>(out.size(), 12);
    if (out.size() > n) {
      out.erase(out.begin() + n, out.end());
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Single case runner — returns checksum + median µs.
// ---------------------------------------------------------------------------

struct CaseResult {
  std::string key;
  const char* stratum;
  double checksum_v;
  double median_us;
  bool ok;
};

// RAII guard for scoped env var override. Restores previous value on
// destruction so the harness can flip MLX_SDPA_FORCE_LEGACY per-call
// without polluting subsequent tests.
struct ScopedEnv {
  std::string name;
  std::string prev;
  bool had_prev;
  ScopedEnv(const char* n, const char* v) : name(n) {
    const char* p = std::getenv(n);
    had_prev = (p != nullptr);
    if (had_prev) prev = p;
    if (v) {
      setenv(n, v, /*overwrite=*/1);
    } else {
      unsetenv(n);
    }
  }
  ~ScopedEnv() {
    if (had_prev) {
      setenv(name.c_str(), prev.c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

CaseResult run_case(const CaseSpec& c, uint64_t seed) {
  CaseResult r{};
  r.key = case_key(c);
  r.stratum = c.label;

  auto q = rand_like(qkv_shape(c.B, c.H_q, c.L_q, c.D), c.dtype, seed + 1);
  auto k = rand_like(qkv_shape(c.B, c.H_k, c.L_k, c.D), c.dtype, seed + 2);
  auto v = rand_like(qkv_shape(c.B, c.H_k, c.L_k, c.D), c.dtype, seed + 3);
  eval(q);
  eval(k);
  eval(v);

  const float scale = 1.0f / std::sqrt(static_cast<float>(c.D));

  auto run = [&]() {
    auto out = fast::scaled_dot_product_attention(
        q, k, v, scale, c.mask_mode,
        /*mask_arr=*/std::nullopt,
        /*sinks=*/std::nullopt);
    eval(out);
    return out;
  };

  // Two runs — check determinism on whatever path ab_enabled()
  // selects.
  auto out_a = run();
  auto out_b = run();

  double cs_a = checksum(out_a);
  double cs_b = checksum(out_b);
  r.checksum_v = cs_a;
  r.ok = (cs_a == cs_b);

  // If AB gate is on, also run the legacy path via the debug
  // override and check numerical equivalence. This is the Phase 1
  // correctness gate: unified ≡ legacy within tolerance.
  if (ab_mode()) {
    array out_legacy = [&]() {
      ScopedEnv force_legacy("MLX_SDPA_FORCE_LEGACY", "1");
      auto o = run();
      return o;
    }();
    auto [rtol, atol] = tolerance_for(c.dtype);
    bool eq = allclose(out_a, out_legacy, rtol, atol).item<bool>();
    if (!eq) {
      r.ok = false;
      MESSAGE("  unified != legacy on ", r.key);
    }
  }

  r.median_us = time_us_median(
      [&]() {
        auto o = run();
        (void)o;
      },
      /*samples=*/5);
  return r;
}

// ---------------------------------------------------------------------------
// Markdown artifact writer
// ---------------------------------------------------------------------------

void write_baseline_artifact(const std::vector<CaseResult>& rows) {
  const std::string path = baseline_out_path();
  std::ofstream f(path);
  REQUIRE(f.good());

  f << "# SDPA Option C — Phase 0 Baseline\n\n";
  f << "Captured on legacy vector-path SDPA (`sdpa_vector` + "
       "`sdpa_vector_2pass`). Phase 1 unified-kernel output must "
       "match the `checksum` column within tolerance; Phase 2 AB "
       "migration must match byte-for-byte. The `median_us` column "
       "is the 1.25× perf floor the Phase 1 kernel must respect "
       "per-case (warning, not hard-fail).\n\n";
  f << "Rows: " << rows.size() << "\n\n";
  f << "| Case key | Stratum | Det? | Checksum | median µs |\n";
  f << "|---|---|---|---:|---:|\n";
  for (const auto& r : rows) {
    f << "| `" << r.key << "` | " << r.stratum << " | "
      << (r.ok ? "✓" : "✗") << " | "
      << std::fixed << std::scientific << r.checksum_v << " | "
      << std::fixed << r.median_us << " |\n";
  }
  f.flush();
  MESSAGE("baseline artifact written to: ", path);
}

} // namespace

// ---------------------------------------------------------------------------
// TEST CASES
// ---------------------------------------------------------------------------

TEST_CASE(
    "sdpa regression: vector-path correctness sweep + baseline capture") {
  if (!sweep_enabled()) {
    MESSAGE("skipping sweep — set MLX_SDPA_SWEEP=1 to opt in");
    return;
  }

  auto sweep = build_sweep();
  MESSAGE("running sweep over ", sweep.size(), " cases");

  std::vector<CaseResult> rows;
  rows.reserve(sweep.size());

  for (size_t i = 0; i < sweep.size(); ++i) {
    const auto& c = sweep[i];
    CAPTURE(i);
    CAPTURE(case_key(c));
    auto r = run_case(c, /*seed=*/0xBADC0FFEEULL + i);
    CHECK_MESSAGE(r.ok, "determinism failure on ", r.key);
    rows.push_back(std::move(r));
  }

  write_baseline_artifact(rows);
}

TEST_CASE("sdpa regression: causal mask isolation") {
  if (!sweep_enabled()) return;

  // Under SDPA's causal convention, Q row i attends to K/V[0..L_k - L_q + i].
  // With L_q == L_k, this collapses to i attending to [0..i]. Poisoning
  // K/V[mid..] must leave output rows [0..mid-1] unchanged.

  const int B = 1, H = 4, D = 64;
  const int L = 16;
  const int mid = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto dt = float16;

  auto q = rand_like({B, H, L, D}, dt, 42);
  auto k = rand_like({B, H, L, D}, dt, 43);
  auto v = rand_like({B, H, L, D}, dt, 44);
  eval(q);
  eval(k);
  eval(v);

  auto out_clean = fast::scaled_dot_product_attention(
      q, k, v, scale, "causal", std::nullopt, std::nullopt);
  eval(out_clean);

  auto k_poison = rand_like({B, H, L, D}, dt, 99);
  auto v_poison = rand_like({B, H, L, D}, dt, 101);

  // k_mix = concat(k[:mid], k_poison[mid:]) along seq axis.
  auto k_head = slice(k, {0, 0, 0, 0}, {B, H, mid, D});
  auto v_head = slice(v, {0, 0, 0, 0}, {B, H, mid, D});
  auto k_ptail = slice(k_poison, {0, 0, mid, 0}, {B, H, L, D});
  auto v_ptail = slice(v_poison, {0, 0, mid, 0}, {B, H, L, D});
  auto k_mix = concatenate({k_head, k_ptail}, /*axis=*/2);
  auto v_mix = concatenate({v_head, v_ptail}, /*axis=*/2);
  eval(k_mix);
  eval(v_mix);

  auto out_poison = fast::scaled_dot_product_attention(
      q, k_mix, v_mix, scale, "causal", std::nullopt, std::nullopt);
  eval(out_poison);

  // Rows [0..mid-1] attend only to K/V[0..mid-1], which are identical
  // between clean and poison. So those rows must match.
  auto clean_head = slice(out_clean, {0, 0, 0, 0}, {B, H, mid, D});
  auto poison_head = slice(out_poison, {0, 0, 0, 0}, {B, H, mid, D});
  CHECK(allclose(clean_head, poison_head, 1e-3, 1e-3).item<bool>());
}

TEST_CASE("sdpa regression: GQA equivalence") {
  if (!sweep_enabled()) return;

  // H_q=8, H_k=1 with broadcast ≡ H_q=H_k=8 with K/V replicated.
  const int B = 1, L_q = 1, L_k = 128, D = 64;
  const int H_q = 8, H_k = 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto dt = float16;

  auto q = rand_like({B, H_q, L_q, D}, dt, 7);
  auto k1 = rand_like({B, H_k, L_k, D}, dt, 8);
  auto v1 = rand_like({B, H_k, L_k, D}, dt, 9);
  eval(q);
  eval(k1);
  eval(v1);

  auto out_gqa = fast::scaled_dot_product_attention(
      q, k1, v1, scale, "causal", std::nullopt, std::nullopt);
  eval(out_gqa);

  // Replicate K/V across H_q heads: broadcast_to + reshape is the
  // equivalent manual expansion.
  auto k8 = broadcast_to(k1, {B, H_q, L_k, D});
  auto v8 = broadcast_to(v1, {B, H_q, L_k, D});
  eval(k8);
  eval(v8);

  auto out_full = fast::scaled_dot_product_attention(
      q, k8, v8, scale, "causal", std::nullopt, std::nullopt);
  eval(out_full);

  CHECK(allclose(out_gqa, out_full, 1e-3, 1e-3).item<bool>());
}

TEST_CASE("sdpa regression: determinism across repeated calls") {
  if (!sweep_enabled()) return;

  const int B = 1, H = 4, D = 64;
  const int L_q = 1, L_k = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  for (auto dt : {float32, float16, bfloat16}) {
    CAPTURE(dtype_str(dt));
    auto q = rand_like({B, H, L_q, D}, dt, 11);
    auto k = rand_like({B, H, L_k, D}, dt, 12);
    auto v = rand_like({B, H, L_k, D}, dt, 13);
    eval(q);
    eval(k);
    eval(v);

    auto o1 = fast::scaled_dot_product_attention(
        q, k, v, scale, "causal", std::nullopt, std::nullopt);
    eval(o1);
    auto o2 = fast::scaled_dot_product_attention(
        q, k, v, scale, "causal", std::nullopt, std::nullopt);
    eval(o2);

    double c1 = checksum(o1);
    double c2 = checksum(o2);
    CHECK(c1 == c2);
  }
}
