// Copyright © 2026 Apple Inc.
//
// Affine-quantized gather-QMV — argument-buffer variants of the
// non-sorted MoE decode matmul path (`gather_qmv` /
// `gather_qmv_fast`). Called from `GatherQMM::eval_gpu` when
// `transpose_ == true` && M < vector_limit, i.e. every decode step
// on a model with affine-quantized MoE experts and non-sorted
// expert indices.
//
// The kernel surface includes variable-length shape/stride arrays
// driving `adjust_matrix_offsets`. We cap each array at
// MAX_NDIMS=4 inline; all production models currently have
// `x_batch_ndims`, `w_batch_ndims`, and gather `batch_ndims`
// <= 3 (typically 1 for MoE decode). The C++ caller asserts the
// cap at runtime.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define AB_GQMV_MAX_NDIMS 4

// Byte layout must match the ArgumentBuffer slot layout declared by
// `gather_qmv()` in quantized.cpp under the AB path. Kept in one
// place here for correspondence with kernel-side struct.
struct BufferPtrOffset {
  uint64_t addr;
  uint64_t offset;
};

struct AffineGatherQmvArgs {
  // Buffer pointers (7 slots × 16 B = 112 B, offset 0..111).
  BufferPtrOffset w;
  BufferPtrOffset scales;
  BufferPtrOffset biases;
  BufferPtrOffset x;
  BufferPtrOffset lhs_indices;
  BufferPtrOffset rhs_indices;
  BufferPtrOffset y;
  // Scalars (offset 112..127).
  int in_vec_size;
  int out_vec_size;
  int x_batch_ndims;
  int w_batch_ndims;
  // Batch (gather) metadata (offset 128..159).
  int batch_ndims;
  int _pad0;        // align the following int64[] at 8-byte boundary
  int64_t lhs_strides[AB_GQMV_MAX_NDIMS]; // 32 B (offset 136..167)
  int64_t rhs_strides[AB_GQMV_MAX_NDIMS]; // 32 B (offset 168..199)
  int batch_shape[AB_GQMV_MAX_NDIMS];     // 16 B (offset 200..215)
  // x shape/strides.
  int _pad1;                              // align int64[] at 8-byte boundary
  int64_t x_strides[AB_GQMV_MAX_NDIMS];
  int x_shape[AB_GQMV_MAX_NDIMS];
  int _pad2;
  // w shape/strides.
  int64_t w_strides[AB_GQMV_MAX_NDIMS];
  int w_shape[AB_GQMV_MAX_NDIMS];
  int _pad3;
  // scales + biases strides.
  int64_t s_strides[AB_GQMV_MAX_NDIMS];
  int64_t b_strides[AB_GQMV_MAX_NDIMS];
};

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv_ab(
    constant const AffineGatherQmvArgs& args [[buffer(0)]],
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
  const device uint32_t* lhs_indices =
      reinterpret_cast<const device uint32_t*>(
          args.lhs_indices.addr + args.lhs_indices.offset);
  const device uint32_t* rhs_indices =
      reinterpret_cast<const device uint32_t*>(
          args.rhs_indices.addr + args.rhs_indices.offset);
  device T* y =
      reinterpret_cast<device T*>(args.y.addr + args.y.offset);

  const constant int& in_vec_size = args.in_vec_size;
  const constant int& out_vec_size = args.out_vec_size;
  const constant int& x_batch_ndims = args.x_batch_ndims;
  const constant int& w_batch_ndims = args.w_batch_ndims;
  const constant int& batch_ndims = args.batch_ndims;

  int M = args.x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x, w, scales, biases, lhs_indices, rhs_indices, y,
      out_vec_size * M,
      batch_ndims, &args.batch_shape[0], &args.lhs_strides[0], &args.rhs_strides[0],
      x_batch_ndims, &args.x_shape[0], &args.x_strides[0],
      w_batch_ndims, &args.w_shape[0], &args.w_strides[0],
      &args.s_strides[0], &args.b_strides[0],
      tid);
  qmv_impl<T, group_size, bits>(
      w, scales, biases, x, y,
      in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv_fast_ab(
    constant const AffineGatherQmvArgs& args [[buffer(0)]],
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
  const device uint32_t* lhs_indices =
      reinterpret_cast<const device uint32_t*>(
          args.lhs_indices.addr + args.lhs_indices.offset);
  const device uint32_t* rhs_indices =
      reinterpret_cast<const device uint32_t*>(
          args.rhs_indices.addr + args.rhs_indices.offset);
  device T* y =
      reinterpret_cast<device T*>(args.y.addr + args.y.offset);

  const constant int& in_vec_size = args.in_vec_size;
  const constant int& out_vec_size = args.out_vec_size;
  const constant int& x_batch_ndims = args.x_batch_ndims;
  const constant int& w_batch_ndims = args.w_batch_ndims;
  const constant int& batch_ndims = args.batch_ndims;

  int M = args.x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x, w, scales, biases, lhs_indices, rhs_indices, y,
      out_vec_size * M,
      batch_ndims, &args.batch_shape[0], &args.lhs_strides[0], &args.rhs_strides[0],
      x_batch_ndims, &args.x_shape[0], &args.x_strides[0],
      w_batch_ndims, &args.w_shape[0], &args.w_strides[0],
      &args.s_strides[0], &args.b_strides[0],
      tid);
  qmv_fast_impl<T, group_size, bits>(
      w, scales, biases, x, y,
      in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

#define instantiate_affine_gather_qmv_ab(name, itype, group_size, bits)       \
  instantiate_kernel(                                                          \
      "affine_gather_qmv_ab_" #name "_gs_" #group_size "_b_" #bits,            \
      affine_gather_qmv_ab, itype, group_size, bits)                           \
  instantiate_kernel(                                                          \
      "affine_gather_qmv_fast_ab_" #name "_gs_" #group_size "_b_" #bits,       \
      affine_gather_qmv_fast_ab, itype, group_size, bits)

#define instantiate_affine_gather_qmv_ab_type(name, itype)            \
  instantiate_affine_gather_qmv_ab(name, itype, 32, 2)                \
  instantiate_affine_gather_qmv_ab(name, itype, 32, 3)                \
  instantiate_affine_gather_qmv_ab(name, itype, 32, 4)                \
  instantiate_affine_gather_qmv_ab(name, itype, 32, 6)                \
  instantiate_affine_gather_qmv_ab(name, itype, 32, 8)                \
  instantiate_affine_gather_qmv_ab(name, itype, 64, 2)                \
  instantiate_affine_gather_qmv_ab(name, itype, 64, 3)                \
  instantiate_affine_gather_qmv_ab(name, itype, 64, 4)                \
  instantiate_affine_gather_qmv_ab(name, itype, 64, 6)                \
  instantiate_affine_gather_qmv_ab(name, itype, 64, 8)                \
  instantiate_affine_gather_qmv_ab(name, itype, 128, 2)               \
  instantiate_affine_gather_qmv_ab(name, itype, 128, 3)               \
  instantiate_affine_gather_qmv_ab(name, itype, 128, 4)               \
  instantiate_affine_gather_qmv_ab(name, itype, 128, 6)               \
  instantiate_affine_gather_qmv_ab(name, itype, 128, 8)

instantiate_affine_gather_qmv_ab_type(float16, half)
instantiate_affine_gather_qmv_ab_type(bfloat16, bfloat16_t)
instantiate_affine_gather_qmv_ab_type(float32, float)
// clang-format on
