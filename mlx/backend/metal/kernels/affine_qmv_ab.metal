// Copyright © 2026 Apple Inc.
//
// Affine-quantized QMV — argument-buffer variant.
//
// Covers the non-batched decode hot-path (B == 1, M == 1, transposed
// weights, affine mode) for `QuantizedMatmul::eval_gpu`. This is the
// dispatch taken by every Q/K/V/O projection and every MLP up/down
// projection on a 4-bit affine-quantized model like Gemma 4 E2B during
// decode — many dispatches per step, so collapsing the 7-binding
// legacy layout into a single AB bind is the highest-leverage
// migration in Phase 2 of the AB adoption plan.
//
// Two kernels: `affine_qmv_ab` (general K/N) and
// `affine_qmv_fast_ab` (aligned K % 512 == 0 && N % 8 == 0 fast path).
// Batched (`batched=1`) kernels stay on the legacy path because they
// need shape/stride arrays that aren't worth plumbing into an AB slot
// layout for this first pass.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

// Byte layout must match the ArgumentBuffer slot ordering declared by
// `qmv` / `qmv_fast` in quantized.cpp under the AB path:
//   [ 0..15] w       : BufferPtrOffset  (device uint32_t*)
//   [16..31] scales  : BufferPtrOffset  (device T*)
//   [32..47] biases  : BufferPtrOffset  (device T*)
//   [48..63] x       : BufferPtrOffset  (device T*)
//   [64..79] y       : BufferPtrOffset  (device T*)
//   [80..83] K       : int   (in_vec_size)
//   [84..87] N       : int   (out_vec_size)
//   [88..95] _pad    : padding to 16-byte multiple
struct BufferPtrOffset {
  uint64_t addr;
  uint64_t offset;
};

struct AffineQmvArgs {
  BufferPtrOffset w;
  BufferPtrOffset scales;
  BufferPtrOffset biases;
  BufferPtrOffset x;
  BufferPtrOffset y;
  int K;
  int N;
  int _pad0;
  int _pad1;
};

template <typename T, int group_size, int bits>
[[kernel]] void affine_qmv_ab(
    constant const AffineQmvArgs& args [[buffer(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const device uint32_t* w =
      reinterpret_cast<const device uint32_t*>(args.w.addr + args.w.offset);
  const device T* scales =
      reinterpret_cast<const device T*>(args.scales.addr + args.scales.offset);
  const device T* biases =
      reinterpret_cast<const device T*>(args.biases.addr + args.biases.offset);
  const device T* x =
      reinterpret_cast<const device T*>(args.x.addr + args.x.offset);
  device T* y =
      reinterpret_cast<device T*>(args.y.addr + args.y.offset);

  qmv_impl<T, group_size, bits>(
      w, scales, biases, x, y, args.K, args.N, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_qmv_fast_ab(
    constant const AffineQmvArgs& args [[buffer(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const device uint32_t* w =
      reinterpret_cast<const device uint32_t*>(args.w.addr + args.w.offset);
  const device T* scales =
      reinterpret_cast<const device T*>(args.scales.addr + args.scales.offset);
  const device T* biases =
      reinterpret_cast<const device T*>(args.biases.addr + args.biases.offset);
  const device T* x =
      reinterpret_cast<const device T*>(args.x.addr + args.x.offset);
  device T* y =
      reinterpret_cast<device T*>(args.y.addr + args.y.offset);

  qmv_fast_impl<T, group_size, bits>(
      w, scales, biases, x, y, args.K, args.N, tid, simd_gid, simd_lid);
}

#define instantiate_affine_qmv_ab(name, itype, group_size, bits)                  \
  instantiate_kernel("affine_qmv_ab_" #name "_gs_" #group_size "_b_" #bits,     \
                     affine_qmv_ab, itype, group_size, bits)                    \
  instantiate_kernel("affine_qmv_fast_ab_" #name "_gs_" #group_size "_b_" #bits,\
                     affine_qmv_fast_ab, itype, group_size, bits)

#define instantiate_affine_qmv_ab_type(name, itype) \
  instantiate_affine_qmv_ab(name, itype, 32, 2)     \
  instantiate_affine_qmv_ab(name, itype, 32, 3)     \
  instantiate_affine_qmv_ab(name, itype, 32, 4)     \
  instantiate_affine_qmv_ab(name, itype, 32, 6)     \
  instantiate_affine_qmv_ab(name, itype, 32, 8)     \
  instantiate_affine_qmv_ab(name, itype, 64, 2)     \
  instantiate_affine_qmv_ab(name, itype, 64, 3)     \
  instantiate_affine_qmv_ab(name, itype, 64, 4)     \
  instantiate_affine_qmv_ab(name, itype, 64, 6)     \
  instantiate_affine_qmv_ab(name, itype, 64, 8)     \
  instantiate_affine_qmv_ab(name, itype, 128, 2)    \
  instantiate_affine_qmv_ab(name, itype, 128, 3)    \
  instantiate_affine_qmv_ab(name, itype, 128, 4)    \
  instantiate_affine_qmv_ab(name, itype, 128, 6)    \
  instantiate_affine_qmv_ab(name, itype, 128, 8)

instantiate_affine_qmv_ab_type(float16, half)
instantiate_affine_qmv_ab_type(bfloat16, bfloat16_t)
instantiate_affine_qmv_ab_type(float32, float)
// clang-format on
