// Copyright © 2026 Apple Inc.
//
// Unified vector-decode SDPA kernel — Phase 1 of Option C.
//
// Goal: replace the CPU-side branch between `sdpa_vector` (single-pass)
// and `sdpa_vector_2pass_1+2` (two-pass) with a single kernel that
// covers the full vector-decode parameter space. This removes the
// segment-topology flip at T_k thresholds (1024 / 4096) that breaks
// ICB replay past the recorded T_k (mlx diagnostic 6f097aa6).
//
// Phase 1 design (this file): the body is a near-copy of legacy
// `sdpa_vector`'s single-pass online-softmax layout. `blocks` is
// accepted as a runtime buffer argument (slot 11) and is a no-op in
// this phase — its purpose is to reserve the slot for Phase 2's
// argument-buffer struct. Phase 1 always computes as if blocks == 1
// and reduces within one threadgroup.
//
// Correctness contract vs legacy: for every (B, H_q, H_k, L_q, L_k,
// D, causal, mask, sinks, dtype) combination in the decode vector
// parameter space, `sdpa_unified_vector` produces output numerically
// equivalent to `sdpa_vector` / `sdpa_vector_2pass` within fp tolerance
// (legacy is the reference — enforced by the Phase 0 regression
// harness).
//
// Perf contract: unified kernel is allowed to be up to 1.25x slower
// than the best of {legacy single-pass, legacy 2-pass} per-case in
// Phase 1 (plan doc §Phase 0.4). The 2-pass path's large-N
// parallelism is sacrificed in exchange for topology stability; the
// perf gap narrows as Phase 2 removes per-call setBytes and Phase 4
// (ICB integration) amortizes encoding cost.

#include <metal_simdgroup>

// Function constants for graph-stable choices — identical to legacy
// `sdpa_vector` so switching the CPU dispatch to call this kernel
// does not change PSO compilation inputs. Slots 20..25 match legacy
// convention.
constant bool u_has_mask [[function_constant(30)]];
constant bool u_query_transposed [[function_constant(31)]];
constant bool u_do_causal [[function_constant(32)]];
constant bool u_bool_mask [[function_constant(33)]];
constant bool u_float_mask [[function_constant(34)]];
constant bool u_has_sinks [[function_constant(35)]];

template <typename T, int D, int V = D>
[[kernel]] void sdpa_unified_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    // Phase 1: accepted but unused. Phase 2 will move this into the
    // AB struct and may use it to parallelize K splits.
    const constant int& blocks [[buffer(11)]],
    const device bool* bmask [[buffer(12), function_constant(u_bool_mask)]],
    const device T* fmask [[buffer(13), function_constant(u_float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(14), function_constant(u_has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(15), function_constant(u_has_mask)]],
    const constant int& mask_head_stride
    [[buffer(16), function_constant(u_has_mask)]],
    const device T* sinks [[buffer(17), function_constant(u_has_sinks)]],
    const constant int& num_q_heads
    [[buffer(18), function_constant(u_has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Reference unused parameter to silence -Wunused until Phase 2.
  (void)blocks;

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions.
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      u_query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (u_bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (u_float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and zero the output accumulator.
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (u_has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // Online softmax across the K sequence — each simdgroup handles a
  // stride-BN subset of K.
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (u_do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (u_bool_mask) {
      use_key = bmask[0];
    } else if (u_float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (u_float_mask) {
        score += static_cast<U>(fmask[0]);
      }
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }
    keys += inner_k_stride;
    values += inner_v_stride;
    if (u_bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (u_float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Merge partial {max, sum, out} across simdgroups via threadgroup
  // memory. Phase 1 merge is always single-threadgroup (blocks == 1
  // effective); Phase 2+ may add cross-TG merge logic.
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 2: Argument Buffer variant of the unified vector-SDPA kernel.
// ---------------------------------------------------------------------------
//
// Same body as `sdpa_unified_vector` but reads every argument from a
// single packed `SdpaUnifiedArgs` struct bound at buffer(0). That
// collapses the 12+ setBuffer / setBytes calls of the non-AB path
// into one `set_buffer(ab, 0)` from the CPU side — which is the whole
// point of Option C for ICB: the setBytes arena goes empty and the
// recorder captures a stable arena across T_k changes.
//
// Byte layout below MUST match the `ArgumentBuffer::Slot` ordering
// declared in `sdpa_vector_unified`'s AB path
// (scaled_dot_product_attention.cpp). Keep them locked in step.

struct BufferPtrOffset {
  uint64_t addr;
  uint64_t offset;
};

struct SdpaUnifiedArgs {
  BufferPtrOffset queries;          // 0
  BufferPtrOffset keys;             // 16
  BufferPtrOffset values;           // 32
  BufferPtrOffset out;              // 48
  BufferPtrOffset mask;             // 64  zero when !u_has_mask
  BufferPtrOffset sinks;            // 80  zero when !u_has_sinks
  uint64_t k_head_stride;           // 96
  uint64_t k_seq_stride;            // 104
  uint64_t v_head_stride;           // 112
  uint64_t v_seq_stride;            // 120
  float scale;                      // 128
  uint gqa_factor;                  // 132
  uint N;                           // 136
  uint blocks;                      // 140  reserved; currently always 1
  uint mask_kv_seq_stride;          // 144  valid when u_has_mask
  uint mask_q_seq_stride;           // 148
  uint mask_head_stride;            // 152
  uint num_q_heads;                 // 156  valid when u_has_sinks
}; // sizeof == 160

template <typename T, int D, int V = D>
[[kernel]] void sdpa_unified_vector_ab(
    constant const SdpaUnifiedArgs& args [[buffer(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  // Dereference AB pointers.
  const device T* queries =
      reinterpret_cast<const device T*>(args.queries.addr + args.queries.offset);
  const device T* keys =
      reinterpret_cast<const device T*>(args.keys.addr + args.keys.offset);
  const device T* values =
      reinterpret_cast<const device T*>(args.values.addr + args.values.offset);
  device T* out = reinterpret_cast<device T*>(args.out.addr + args.out.offset);

  const device bool* bmask = nullptr;
  const device T* fmask = nullptr;
  if (u_bool_mask) {
    bmask = reinterpret_cast<const device bool*>(args.mask.addr + args.mask.offset);
  }
  if (u_float_mask) {
    fmask = reinterpret_cast<const device T*>(args.mask.addr + args.mask.offset);
  }
  const device T* sinks_ptr = nullptr;
  if (u_has_sinks) {
    sinks_ptr = reinterpret_cast<const device T*>(args.sinks.addr + args.sinks.offset);
  }

  const int gqa_factor = int(args.gqa_factor);
  const int N = int(args.N);
  const size_t k_head_stride = size_t(args.k_head_stride);
  const size_t k_seq_stride = size_t(args.k_seq_stride);
  const size_t v_head_stride = size_t(args.v_head_stride);
  const size_t v_seq_stride = size_t(args.v_seq_stride);
  const float scale = args.scale;
  const int mask_kv_seq_stride = int(args.mask_kv_seq_stride);
  const int mask_q_seq_stride = int(args.mask_q_seq_stride);
  const int mask_head_stride = int(args.mask_head_stride);
  const int num_q_heads = int(args.num_q_heads);

  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      u_query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (u_bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (u_float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (u_has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks_ptr[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (u_do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (u_bool_mask) {
      use_key = bmask[0];
    } else if (u_float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (u_float_mask) {
        score += static_cast<U>(fmask[0]);
      }
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }
    keys += inner_k_stride;
    values += inner_v_stride;
    if (u_bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (u_float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
