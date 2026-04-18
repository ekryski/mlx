// Copyright © 2026 Apple Inc.
//
// RMSNorm — argument-buffer variant. Replaces the 6 individual
// setBuffer/setBytes bindings used by `rms_norm.metal` with a single
// packed argument struct bound at buffer(0). Layout matches the C++
// `mlx::core::metal::ArgumentBuffer` slot ordering declared in
// `normalization.cpp` for these kernels.
//
// See benchmarks/notes/ab-rmsnorm-pilot-2026-04-17.md for the design.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Byte layout (must match ArgumentBuffer packing in
// RMSNorm::eval_gpu's AB path):
//   [ 0..15] x   : BufferPtrOffset { uint64_t addr; uint64_t offset; }
//   [16..31] w   : BufferPtrOffset
//   [32..47] out : BufferPtrOffset
//   [48..51] eps       : float
//   [52..55] axis_size : uint32
//   [56..59] w_stride  : uint32
//   [60..63] _pad      : uint32   (round to 16-byte alignment)
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

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_ab_single_row(
    constant const RmsArgs& args [[buffer(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  const device T* x =
      reinterpret_cast<const device T*>(args.x.addr + args.x.offset);
  const device T* w =
      reinterpret_cast<const device T*>(args.w.addr + args.w.offset);
  device T* out =
      reinterpret_cast<device T*>(args.out.addr + args.out.offset);
  const float eps = args.eps;
  const uint axis_size = args.axis_size;
  const uint w_stride = args.w_stride;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      float xi = x[i];
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_ab_looped(
    constant const RmsArgs& args [[buffer(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  const device T* x =
      reinterpret_cast<const device T*>(args.x.addr + args.x.offset);
  const device T* w =
      reinterpret_cast<const device T*>(args.w.addr + args.w.offset);
  device T* out =
      reinterpret_cast<device T*>(args.out.addr + args.out.offset);
  const float eps = args.eps;
  const uint axis_size = args.axis_size;
  const uint w_stride = args.w_stride;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          acc += xi * xi;
        }
      }
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[r + i] = w[w_stride * (i + r)] *
            static_cast<T>(x[r + i] * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          out[r + i] = w[w_stride * (i + r)] *
              static_cast<T>(x[r + i] * local_inv_mean[0]);
        }
      }
    }
  }
}

// clang-format off
#define instantiate_rms_ab(name, itype)                             \
  instantiate_kernel("rms_ab" #name, rms_ab_single_row, itype)      \
  instantiate_kernel("rms_ab_looped" #name, rms_ab_looped, itype)

instantiate_rms_ab(float32, float)
instantiate_rms_ab(float16, half)
instantiate_rms_ab(bfloat16, bfloat16_t) // clang-format on
