// Copyright © 2026 Eric Kryski. TurboFlash Pass 1 kernels for compressed-domain attention.
//
// These are the hottest TurboQuant kernels — run every decode token.
// Runtime-varying params (token_count, num_blocks, repeat_count, BlockSize)
// are buffer arguments to avoid per-token pipeline recompilation.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// ============================================================================
// TurboFlash Pass 1: Standard (non-causal, L=1 decode)
// ============================================================================
template <int KeyBits, int ValueBits, int Dim, int KeyPackedWidth, int ValuePackedWidth>
[[kernel]] void turbo_flash_p1(
    const device float* q_rot [[buffer(0)]],
    const device uint32_t* key_packed [[buffer(1)]],
    const device float* key_norms [[buffer(2)]],
    const device float* key_codebook [[buffer(3)]],
    const device uint32_t* val_packed [[buffer(4)]],
    const device float* val_norms [[buffer(5)]],
    const device float* val_codebook [[buffer(6)]],
    device float* o_partials [[buffer(7)]],
    device float* m_partials [[buffer(8)]],
    device float* l_partials [[buffer(9)]],
    constant int& token_count [[buffer(10)]],
    constant int& repeat_count [[buffer(11)]],
    constant int& num_blocks [[buffer(12)]],
    constant int& block_size [[buffer(13)]],
    uint3 pos [[thread_position_in_grid]]) {

  constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
  constexpr uint KEY_LEVELS = 1u << KeyBits;
  constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
  constexpr uint VAL_LEVELS = 1u << ValueBits;
  constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

  uint lane = pos.x;
  uint q_idx = pos.y;
  uint block_idx = pos.z;
  uint kv_idx = q_idx / uint(repeat_count);

  uint t_start = block_idx * uint(block_size);
  uint t_end = t_start + uint(block_size);
  if (t_end > uint(token_count)) t_end = uint(token_count);

  float key_cb[KEY_LEVELS];
  for (uint i = 0; i < KEY_LEVELS; i++) key_cb[i] = key_codebook[i];
  float val_cb[VAL_LEVELS];
  for (uint i = 0; i < VAL_LEVELS; i++) val_cb[i] = val_codebook[i];

  float q_vals[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    q_vals[i] = (d < uint(Dim)) ? q_rot[q_idx * Dim + d] : 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  float o[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

  for (uint t = t_start; t < t_end; t++) {
    const device uint32_t* k_packed_ptr = key_packed + kv_idx * uint(token_count) * KeyPackedWidth + t * KeyPackedWidth;
    float k_norm = key_norms[kv_idx * uint(token_count) + t];

    float dot_partial = 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= uint(Dim)) break;
      uint k_bit_offset = d * KeyBits;
      uint k_word_idx = k_bit_offset / 32;
      uint k_shift = k_bit_offset % 32;
      uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
      int k_spill = (int)k_shift + (int)KeyBits - 32;
      if (k_spill > 0) k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
      k_value &= KEY_MASK;
      dot_partial += q_vals[i] * key_cb[k_value];
    }
    float score = simd_sum(dot_partial) * k_norm;

    float new_m = max(m, score);
    float exp_diff = exp(m - new_m);
    float exp_score = exp(score - new_m);

    const device uint32_t* v_packed_ptr = val_packed + kv_idx * uint(token_count) * ValuePackedWidth + t * ValuePackedWidth;
    float v_norm = val_norms[kv_idx * uint(token_count) + t];

    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= uint(Dim)) break;
      uint v_bit_offset = d * ValueBits;
      uint v_word_idx = v_bit_offset / 32;
      uint v_shift = v_bit_offset % 32;
      uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
      int v_spill = (int)v_shift + (int)ValueBits - 32;
      if (v_spill > 0) v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
      v_value &= VAL_MASK;
      o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
    }
    l = l * exp_diff + exp_score;
    m = new_m;
  }

  uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    if (d < uint(Dim)) o_partials[partial_base + d] = o[i];
  }
  if (lane == 0) {
    uint ml_idx = q_idx * uint(num_blocks) + block_idx;
    m_partials[ml_idx] = m;
    l_partials[ml_idx] = l;
  }
}

// ============================================================================
// TurboFlash Pass 1 Causal: With per-query causal masking for L>1 prefill
// ============================================================================
template <int KeyBits, int ValueBits, int Dim, int KeyPackedWidth, int ValuePackedWidth>
[[kernel]] void turbo_flash_p1_causal(
    const device float* q_rot [[buffer(0)]],
    const device uint32_t* key_packed [[buffer(1)]],
    const device float* key_norms [[buffer(2)]],
    const device float* key_codebook [[buffer(3)]],
    const device uint32_t* val_packed [[buffer(4)]],
    const device float* val_norms [[buffer(5)]],
    const device float* val_codebook [[buffer(6)]],
    device float* o_partials [[buffer(7)]],
    device float* m_partials [[buffer(8)]],
    device float* l_partials [[buffer(9)]],
    constant int& token_count [[buffer(10)]],
    constant int& repeat_count [[buffer(11)]],
    constant int& num_blocks [[buffer(12)]],
    constant int& block_size [[buffer(13)]],
    constant int& L [[buffer(14)]],
    constant int& q_offset [[buffer(15)]],
    uint3 pos [[thread_position_in_grid]]) {

  constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
  constexpr uint KEY_LEVELS = 1u << KeyBits;
  constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
  constexpr uint VAL_LEVELS = 1u << ValueBits;
  constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

  uint lane = pos.x;
  uint q_idx = pos.y;
  uint block_idx = pos.z;

  uint q_within_L = q_idx % uint(L);
  uint q_head_idx = q_idx / uint(L);
  uint kv_idx = q_head_idx / uint(repeat_count);
  uint q_abs = uint(q_offset) + q_within_L;

  uint t_start = block_idx * uint(block_size);
  uint t_end = t_start + uint(block_size);
  if (t_end > uint(token_count)) t_end = uint(token_count);

  // Early exit: entire block is future-masked
  if (t_start > q_abs) {
    uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d < uint(Dim)) o_partials[partial_base + d] = 0.0f;
    }
    if (lane == 0) {
      uint ml_idx = q_idx * uint(num_blocks) + block_idx;
      m_partials[ml_idx] = -INFINITY;
      l_partials[ml_idx] = 0.0f;
    }
    return;
  }
  if (t_end > q_abs + 1) t_end = q_abs + 1;

  float key_cb[KEY_LEVELS];
  for (uint i = 0; i < KEY_LEVELS; i++) key_cb[i] = key_codebook[i];
  float val_cb[VAL_LEVELS];
  for (uint i = 0; i < VAL_LEVELS; i++) val_cb[i] = val_codebook[i];

  float q_vals[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    q_vals[i] = (d < uint(Dim)) ? q_rot[q_idx * Dim + d] : 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  float o[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

  for (uint t = t_start; t < t_end; t++) {
    const device uint32_t* k_packed_ptr = key_packed + kv_idx * uint(token_count) * KeyPackedWidth + t * KeyPackedWidth;
    float k_norm = key_norms[kv_idx * uint(token_count) + t];

    float dot_partial = 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= uint(Dim)) break;
      uint k_bit_offset = d * KeyBits;
      uint k_word_idx = k_bit_offset / 32;
      uint k_shift = k_bit_offset % 32;
      uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
      int k_spill = (int)k_shift + (int)KeyBits - 32;
      if (k_spill > 0) k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
      k_value &= KEY_MASK;
      dot_partial += q_vals[i] * key_cb[k_value];
    }
    float score = simd_sum(dot_partial) * k_norm;

    float new_m = max(m, score);
    float exp_diff = exp(m - new_m);
    float exp_score = exp(score - new_m);

    const device uint32_t* v_packed_ptr = val_packed + kv_idx * uint(token_count) * ValuePackedWidth + t * ValuePackedWidth;
    float v_norm = val_norms[kv_idx * uint(token_count) + t];

    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= uint(Dim)) break;
      uint v_bit_offset = d * ValueBits;
      uint v_word_idx = v_bit_offset / 32;
      uint v_shift = v_bit_offset % 32;
      uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
      int v_spill = (int)v_shift + (int)ValueBits - 32;
      if (v_spill > 0) v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
      v_value &= VAL_MASK;
      o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
    }
    l = l * exp_diff + exp_score;
    m = new_m;
  }

  uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    if (d < uint(Dim)) o_partials[partial_base + d] = o[i];
  }
  if (lane == 0) {
    uint ml_idx = q_idx * uint(num_blocks) + block_idx;
    m_partials[ml_idx] = m;
    l_partials[ml_idx] = l;
  }
}

// ============================================================================
// Instantiation macros for TurboFlash Pass 1
// ============================================================================
#define TQ_PW(dim, bits) (((dim) * (bits) + 31) / 32)

#define instantiate_turbo_flash_p1(kb, vb, dim) \
  template [[host_name("turbo_flash_p1_" #kb "_" #vb "_" #dim)]] \
  [[kernel]] void turbo_flash_p1<kb, vb, dim, TQ_PW(dim, kb), TQ_PW(dim, vb)>( \
    const device float*, const device uint32_t*, const device float*, \
    const device float*, const device uint32_t*, const device float*, \
    const device float*, device float*, device float*, device float*, \
    constant int&, constant int&, constant int&, constant int&, uint3); \
  template [[host_name("turbo_flash_p1_causal_" #kb "_" #vb "_" #dim)]] \
  [[kernel]] void turbo_flash_p1_causal<kb, vb, dim, TQ_PW(dim, kb), TQ_PW(dim, vb)>( \
    const device float*, const device uint32_t*, const device float*, \
    const device float*, const device uint32_t*, const device float*, \
    const device float*, device float*, device float*, device float*, \
    constant int&, constant int&, constant int&, constant int&, \
    constant int&, constant int&, uint3);

// Common asymmetric (KeyBits, ValueBits) combinations
#define instantiate_flash_for_dim(dim) \
  instantiate_turbo_flash_p1(4, 4, dim) \
  instantiate_turbo_flash_p1(4, 2, dim) \
  instantiate_turbo_flash_p1(4, 3, dim) \
  instantiate_turbo_flash_p1(3, 2, dim) \
  instantiate_turbo_flash_p1(3, 3, dim) \
  instantiate_turbo_flash_p1(8, 4, dim) \
  instantiate_turbo_flash_p1(8, 2, dim) \
  instantiate_turbo_flash_p1(8, 8, dim)

instantiate_flash_for_dim(64)
instantiate_flash_for_dim(80)
instantiate_flash_for_dim(96)
instantiate_flash_for_dim(128)
instantiate_flash_for_dim(256)
