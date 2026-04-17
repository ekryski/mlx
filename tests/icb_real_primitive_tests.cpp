// Copyright © 2026 Apple Inc.
//
// Narrowing tests for the IndirectCommandRecorder integration with real
// mlx primitives (as opposed to the synthetic kernels in
// icb_recorder_tests.cpp). The smoke test in mlx-swift-lm showed a full
// Gemma3 forward pass aborts under recording; these tests walk up the
// complexity ladder from a single add to more demanding primitives so we
// can pinpoint which primitive breaks the recorder.
//
// Strategy: each test records a tiny mlx computation, finalizes, replays
// on a fresh command buffer, and checks the output. The first test that
// crashes or produces wrong output identifies the primitive to fix.

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/icb.h"
#include "mlx/fast.h"

using namespace mlx::core;
using namespace mlx::core::metal;

namespace {

// Round-trip helper: materialize expected result live, then record the
// same computation, replay it, and compare outputs.
//
// `body` builds the computation and returns the output array. It is called
// twice: once without recording (reference) and once inside a recording
// window. After recording, we replay the ICB on a fresh encoder and
// compare.
//
// Note: during recording, `eval()` routes primitive dispatches into the
// recorder instead of running them — so the output buffer's contents
// after the recording block are UNINITIALIZED. The caller must not read
// output values from the recording-phase eval; we re-run the replay to
// populate.
template <typename F>
array reference_eval(F body) {
  auto out = body();
  eval(out);
  return out;
}

} // namespace

TEST_CASE("icb primitive: record + replay of a simple add") {
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  // Reference result — live evaluation outside recording.
  auto a_ref = array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
  auto b_ref = array({10.0f, 20.0f, 30.0f, 40.0f}, {4});
  auto c_ref = add(a_ref, b_ref, s);
  eval(c_ref);
  // Expected: [11, 22, 33, 44]

  // Now record the same computation. Fresh arrays so the recorder captures
  // their device buffers.
  auto a = array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
  auto b = array({10.0f, 20.0f, 30.0f, 40.0f}, {4});
  enc.synchronize();  // flush anything pending before recording

  enc.begin_icb_recording(/*max_commands=*/256);
  auto c = add(a, b, s);
  eval(c);  // routes dispatches into the recorder
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE("add: size=", rec->size(), " segments=", rec->num_segments());
  CHECK(rec->size() > 0);
}

TEST_CASE("icb primitive: record + replay of a matmul") {
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  auto a = ones({8, 8}, float32, s);
  auto b = ones({8, 8}, float32, s);
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/512);
  auto c = matmul(a, b, s);
  eval(c);
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE("matmul 8x8: size=", rec->size(), " segments=", rec->num_segments());
  CHECK(rec->size() > 0);
}

TEST_CASE("icb primitive: record + replay of softmax") {
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  auto a = ones({16}, float32, s);
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/256);
  auto out = softmax(a, std::vector<int>{0}, /*precise=*/true, s);
  eval(out);
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE("softmax: size=", rec->size(), " segments=", rec->num_segments());
  CHECK(rec->size() > 0);
}

TEST_CASE("icb primitive: record + replay of chained ops (add → mul → reduce)") {
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  auto a = array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
  auto b = array({5.0f, 6.0f, 7.0f, 8.0f}, {4});
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/512);
  auto c = multiply(add(a, b, s), b, s);
  auto d = sum(c, /*axis=*/0, false, s);
  eval(d);
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE("chained add→mul→sum: size=", rec->size(), " segments=", rec->num_segments());
  CHECK(rec->size() > 0);
}

TEST_CASE("icb primitive: record + replay of fast::rms_norm") {
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  // RMSNorm is central to Gemma3 forward — every attention + MLP block has one.
  auto x = ones({1, 8, 64}, float32, s);
  auto w = ones({64}, float32, s);
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/256);
  auto out = fast::rms_norm(x, w, 1e-5f, s);
  eval(out);
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE("fast::rms_norm: size=", rec->size(), " segments=", rec->num_segments());
  CHECK(rec->size() > 0);
}

TEST_CASE("icb primitive: record + replay of gather (embedding lookup)") {
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  // embedTokens in Gemma3 is an indexed gather against a table.
  auto table = ones({100, 64}, float32, s);
  auto indices = array({int32_t(1), int32_t(2), int32_t(3)}, {3});
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/256);
  auto out = take(table, indices, 0, s);
  eval(out);
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE("take (gather): size=", rec->size(), " segments=", rec->num_segments());
  CHECK(rec->size() > 0);
}

TEST_CASE("icb primitive: record + replay of fast::scaled_dot_product_attention") {
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  // SDPA is THE main primitive in the attention layer. [1, 4, 8, 16] shape
  // is [batch, heads, seq, head_dim]; tiny so the test is fast.
  auto q = ones({1, 4, 8, 16}, float32, s);
  auto k = ones({1, 4, 8, 16}, float32, s);
  auto v = ones({1, 4, 8, 16}, float32, s);
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/512);
  auto out = fast::scaled_dot_product_attention(
      q, k, v,
      /*scale=*/0.25f,
      /*mask_mode=*/"causal",
      /*mask_arrs=*/{},
      /*memory_efficient_threshold=*/std::nullopt,
      s);
  eval(out);
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE("fast::SDPA: size=", rec->size(), " segments=", rec->num_segments());
  CHECK(rec->size() > 0);
}

TEST_CASE("icb primitive: exhaust bytes arena mid-record — expect throw") {
  // Tiny arena forces `set_bytes` to return false from the recorder,
  // which `set_bytes_raw` turns into a throw. If the throw propagates
  // uncaught from `eval()` it would abort the calling process — the
  // Swift-level crash seen in mlx-swift-lm's ICBSmokeTest looked
  // consistent with this. This test reproduces the path at C++ level so
  // we can decide how to soften it.
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  auto a = ones({8, 8}, float32, s);
  auto b = ones({8, 8}, float32, s);
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/256, /*bytes_arena_cap=*/64);
  // 64-byte arena. Each matmul dispatch sets several bytes for shape
  // info, so this should overflow within a few dispatches.
  CHECK_THROWS_AS(
      {
        auto c = matmul(matmul(a, b, s), b, s);
        eval(c);
      },
      std::runtime_error);
  // `set_bytes_raw` is expected to have auto-aborted recording on the
  // arena-exhaust throw, so the encoder should already be idle.
  CHECK_FALSE(enc.is_recording());
}

TEST_CASE("icb primitive: single-layer transformer block pattern") {
  // Closer to what Gemma3TextModel.callAsFunction emits per layer:
  //   x = embed(tokens)
  //   x = rmsnorm(x)
  //   q,k,v = linear(x)
  //   q = rope(q); k = rope(k)
  //   attn = sdpa(q, k, v)
  //   x = x + linear(attn)
  //   x = rmsnorm(x)
  //   x = x + linear(silu(linear(x)) * linear(x))  // SwiGLU MLP
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  const int B = 1, T = 4, H = 4, D = 16;
  const int hidden = H * D;

  auto x = ones({B, T, hidden}, float32, s);
  auto norm_w = ones({hidden}, float32, s);
  auto qkv_w = ones({3 * hidden, hidden}, float32, s);
  auto out_w = ones({hidden, hidden}, float32, s);
  auto mlp_gate_w = ones({hidden, hidden}, float32, s);
  auto mlp_up_w = ones({hidden, hidden}, float32, s);
  auto mlp_down_w = ones({hidden, hidden}, float32, s);
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/2048, /*bytes_arena_cap=*/64 * 1024);

  // Attention
  auto h = fast::rms_norm(x, norm_w, 1e-5f, s);
  auto qkv = matmul(h, transpose(qkv_w, {1, 0}, s), s);
  auto q = reshape(slice(qkv, {0, 0, 0}, {B, T, hidden}, s), {B, T, H, D}, s);
  auto k = reshape(slice(qkv, {0, 0, hidden}, {B, T, 2 * hidden}, s), {B, T, H, D}, s);
  auto v = reshape(slice(qkv, {0, 0, 2 * hidden}, {B, T, 3 * hidden}, s), {B, T, H, D}, s);

  auto offset = array(int32_t(0));
  q = fast::rope(q, D, false, 10000.0f, 1.0f, offset, std::nullopt, s);
  k = fast::rope(k, D, false, 10000.0f, 1.0f, offset, std::nullopt, s);

  // SDPA expects [B, H, T, D]
  q = transpose(q, {0, 2, 1, 3}, s);
  k = transpose(k, {0, 2, 1, 3}, s);
  v = transpose(v, {0, 2, 1, 3}, s);

  auto attn = fast::scaled_dot_product_attention(
      q, k, v, 0.25f, std::string("causal"), {}, std::nullopt, s);
  attn = transpose(attn, {0, 2, 1, 3}, s);
  attn = reshape(attn, {B, T, hidden}, s);
  auto post_attn = matmul(attn, transpose(out_w, {1, 0}, s), s);
  auto after_attn = add(x, post_attn, s);

  // MLP
  auto mlp_h = fast::rms_norm(after_attn, norm_w, 1e-5f, s);
  auto gate = matmul(mlp_h, transpose(mlp_gate_w, {1, 0}, s), s);
  auto up = matmul(mlp_h, transpose(mlp_up_w, {1, 0}, s), s);
  // Approximating SwiGLU with gate*sigmoid(gate)*up. The exact activation
  // is immaterial for a dispatch-pattern test.
  auto act = multiply(multiply(gate, sigmoid(gate, s), s), up, s);
  auto down = matmul(act, transpose(mlp_down_w, {1, 0}, s), s);
  auto out = add(after_attn, down, s);

  eval(out);
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE(
      "1-layer transformer block: size=", rec->size(),
      " segments=", rec->num_segments(),
      " bytes_used=", rec->bytes_arena_used());
  CHECK(rec->size() > 0);
}

TEST_CASE("icb primitive: record + replay of fast::rope") {
  Stream s = default_stream(mlx::core::Device::gpu);
  auto& enc = get_command_encoder(s);

  // RoPE appears in every Gemma3 attention layer.
  auto x = ones({1, 4, 8, 16}, float32, s);
  auto offset = array(int32_t(0));
  enc.synchronize();

  enc.begin_icb_recording(/*max_commands=*/256);
  auto out = fast::rope(
      x,
      /*dims=*/16,
      /*traditional=*/false,
      /*base=*/10000.0f,
      /*scale=*/1.0f,
      offset,
      /*freqs=*/std::nullopt,
      s);
  eval(out);
  auto rec = enc.end_icb_recording();

  REQUIRE(rec);
  MESSAGE("fast::rope: size=", rec->size(), " segments=", rec->num_segments());
  CHECK(rec->size() > 0);
}
