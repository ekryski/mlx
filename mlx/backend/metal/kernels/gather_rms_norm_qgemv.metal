// Copyright © 2026 Eric Kryski.
//
// Fused RMSNorm + gathered quantized GEMV for MoE decode (T=1).
// Collapses:
//   1. rms_norm(x, norm_weight, eps)
//   2. gatherQuantizedMM(normed, w, scales, biases, rhsIndices)
// into a single Metal dispatch per MoE expert slot.
//
// x:            [B,        1, 1, K]
// norm_weight:  [K]
// w:            [E,        N, K / pack_factor]   (4-bit packed)
// scales:       [E,        N, K / group_size]
// biases:       [E,        N, K / group_size]
// indices:      [B,        top_k]                (int32, expert id per slot)
// out:          [B,  top_k, 1, N]
//
// Each threadgroup handles one (batch, top_k-slot) pair and produces
// `num_simdgroups * results_per_simdgroup` outputs. RMS inv_rms is
// recomputed per threadgroup — redundant across slots of the same batch,
// but each such compute is ~1 reduction of a K-vector, negligible vs the
// qmatmul work.

#include <metal_common>
#include <metal_simdgroup>

#include "utils.h"

using namespace metal;

// 4-bit qdot (inlined from rms_norm_qgemv).
template <int values_per_thread>
inline float gather_qdot_4bit(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i]     * float(ws[i] & 0x000f) +
         x_thread[4 * i + 1] * float(ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * float(ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * float(ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

template <int values_per_thread>
inline float gather_qdot_4bit_safe(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int remaining) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    int base = 4 * i;
    if (base < remaining) accum += x_thread[base] * float(ws[i] & 0x000f);
    if (base + 1 < remaining) accum += x_thread[base + 1] * float(ws[i] & 0x00f0);
    if (base + 2 < remaining) accum += x_thread[base + 2] * float(ws[i] & 0x0f00);
    if (base + 3 < remaining) accum += x_thread[base + 3] * float(ws[i] & 0xf000);
  }
  return scale * accum + sum * bias;
}

// Fused: RMSNorm + gatherQuantizedMM (4-bit only).
template <typename T, int group_size>
[[kernel]] void gather_rms_norm_qgemv(
    const device T* x [[buffer(0)]],
    const device T* norm_weight [[buffer(1)]],
    const device uint32_t* w [[buffer(2)]],
    const device T* scales [[buffer(3)]],
    const device T* biases [[buffer(4)]],
    const device int32_t* indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant int& in_vec_size [[buffer(8)]],
    constant int& out_vec_size [[buffer(9)]],
    constant int& top_k [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int SIMD_SIZE = 32;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = 8;
  constexpr int values_per_thread = pack_factor;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_pack = 4;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  // tid.z is the global slot index (batch * top_k + k).
  int slot = int(tid.z);
  int batch = slot / top_k;
  // Read expert id. sortedIndices in the switch path may have reordered
  // these, but rhsIndices is already in slot order at the kernel boundary.
  int expert = indices[slot];

  // Offset x and y by batch / slot.
  x += batch * in_vec_size;
  y += slot * out_vec_size;

  // Per-expert stride into w / scales / biases.
  int w_stride_e = out_vec_size * (in_vec_size / pack_factor);
  int sb_stride_e = out_vec_size * (in_vec_size / group_size);
  const device uint8_t* ws = (const device uint8_t*)(w + expert * w_stride_e);
  scales += expert * sb_stride_e;
  biases += expert * sb_stride_e;

  // ======================================================================
  // Phase 1: Load x into shared memory + compute inv_rms.
  // ======================================================================
  threadgroup T shared_x[8192];
  threadgroup float shared_inv_rms[1];

  uint total_threads = num_simdgroups * SIMD_SIZE;
  uint thread_id = simd_gid * SIMD_SIZE + simd_lid;

  for (uint i = thread_id; i < uint(in_vec_size); i += total_threads) {
    shared_x[i] = x[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float sum_sq = 0;
  for (uint i = thread_id; i < uint(in_vec_size); i += total_threads) {
    float v = float(shared_x[i]);
    sum_sq += v * v;
  }
  sum_sq = simd_sum(sum_sq);

  threadgroup float simd_sums[2];
  if (simd_lid == 0) simd_sums[simd_gid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0 && simd_lid == 0) {
    float total = simd_sums[0] + simd_sums[1];
    shared_inv_rms[0] = metal::precise::rsqrt(total / float(in_vec_size) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float inv_rms = shared_inv_rms[0];

  // ======================================================================
  // Phase 2: Quantized GEMV on normed-x vs expert-specific weights.
  // ======================================================================
  thread float x_thread[values_per_thread];
  thread float result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = int(tid.y) * (num_simdgroups * results_per_simdgroup) +
      int(simd_gid) * results_per_simdgroup;

  if (out_row >= out_vec_size) return;

  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  ws += used_out_row * in_vec_size_w + simd_lid * bytes_per_pack;
  scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  y += used_out_row;

  uint x_offset = simd_lid * values_per_thread;
  int k = 0;

  constexpr float qdot_prescale[8] = {
      1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f,
      1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f
  };

  for (; k < in_vec_size - block_size; k += block_size) {
    float sum = 0;
    for (int i = 0; i < values_per_thread; i++) {
      float raw = float(shared_x[x_offset + i]);
      float nw = float(norm_weight[x_offset + i]);
      float normed = raw * nw * inv_rms;
      sum += normed;
      x_thread[i] = normed * qdot_prescale[i];
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      float s = float(scales[row * in_vec_size_g]);
      float b = float(biases[row * in_vec_size_g]);
      result[row] += gather_qdot_4bit<values_per_thread>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x_offset += block_size;
  }

  const int remaining = clamp(
      int(in_vec_size) - k - int(simd_lid * values_per_thread),
      0, values_per_thread);
  if (remaining > 0) {
    float sum = 0;
    for (int i = 0; i < values_per_thread; i++) {
      if (i < remaining) {
        float raw = float(shared_x[x_offset + i]);
        float nw = float(norm_weight[x_offset + i]);
        float normed = raw * nw * inv_rms;
        sum += normed;
        x_thread[i] = normed * qdot_prescale[i];
      } else {
        x_thread[i] = 0;
      }
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      float s = float(scales[row * in_vec_size_g]);
      float b = float(biases[row * in_vec_size_g]);
      result[row] += gather_qdot_4bit_safe<values_per_thread>(
          wl, x_thread, s, b, sum, remaining);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && used_out_row + row < out_vec_size) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

#define instantiate_gather_rms_norm_qgemv(type, tname, gs) \
  template [[host_name("gather_rms_norm_qgemv_" #tname "_gs" #gs)]] \
  [[kernel]] void gather_rms_norm_qgemv<type, gs>( \
      const device type*, const device type*, const device uint32_t*, \
      const device type*, const device type*, const device int32_t*, \
      device type*, constant float&, \
      constant int&, constant int&, constant int&, \
      uint3, uint, uint);

instantiate_gather_rms_norm_qgemv(half, float16, 64)
instantiate_gather_rms_norm_qgemv(bfloat16_t, bfloat16, 64)
instantiate_gather_rms_norm_qgemv(half, float16, 128)
instantiate_gather_rms_norm_qgemv(bfloat16_t, bfloat16, 128)
