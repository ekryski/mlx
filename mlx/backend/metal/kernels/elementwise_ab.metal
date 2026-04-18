// Copyright © 2026 Apple Inc.
//
// Elementwise AB — argument-buffer variants of the AOT-compiled
// vector/vector binary kernels and vector unary kernels most
// frequently called during decode:
//   - binary_vv / binary_vv2 / binary_vvn for Add and Multiply
//     (residual adds, SwiGLU gate-mul, scaling)
//   - unary_v / unary_v2 / unary_vn for Sigmoid, Negative, Abs,
//     Exp (activations + sign-flipping + the softmax prologue)
//
// Other binary variants (ss / sv / vs / g*) and other ops stay on
// legacy. The AOT vv path is ~15 % of per-step dispatches per the
// GPT-OSS-20B E1 dispatch audit.
//
// Historical note: unary was initially blocked because unary_ops.h
// / erf.h / expm1f.h emitted non-inline function definitions for
// ArcCos / ArcSin / ArcTan (complex64 overloads) and erff /
// erfinvf / expm1ff. Sibling .metal files including the same
// headers duplicated those symbols at metallib link. Fixed by
// marking the bodies `inline` in the shared headers; the AOT
// unary.air + this file now coexist in one metallib cleanly.

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/unary_ops.h"

// Byte layout must match the ArgumentBuffer slot layout in the C++
// AB branches (binary.cpp / unary.cpp).
//
// Binary (vv / vv2 / vvn):
//   [ 0..15] a   : BufferPtrOffset
//   [16..31] b   : BufferPtrOffset
//   [32..47] c   : BufferPtrOffset
//   [48..55] size: uint32 (vv/vvn) or int64 (vv2); int64 slot covers both.
//                  Readers interpret the low 32 bits when size fits in uint32.
//
// Unary (v / v2 / vn):
//   [ 0..15] a   : BufferPtrOffset
//   [16..31] c   : BufferPtrOffset
//   [32..39] size: int64 (v2) or uint32 (v/vn) — int64 slot, low bits read.
struct BufferPtrOffset {
  uint64_t addr;
  uint64_t offset;
};

struct BinaryVVArgs {
  BufferPtrOffset a;
  BufferPtrOffset b;
  BufferPtrOffset c;
  int64_t size;
};

struct UnaryVArgs {
  BufferPtrOffset a;
  BufferPtrOffset c;
  int64_t size;
};

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void binary_vv_ab(
    constant const BinaryVVArgs& args [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  const device T* a =
      reinterpret_cast<const device T*>(args.a.addr + args.a.offset);
  const device T* b =
      reinterpret_cast<const device T*>(args.b.addr + args.b.offset);
  device U* c = reinterpret_cast<device U*>(args.c.addr + args.c.offset);
  const uint size = static_cast<uint>(args.size);

  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      c[index + i] = Op()(a[index + i], b[index + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[index + i] = Op()(a[index + i], b[index + i]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void binary_vv2_ab(
    constant const BinaryVVArgs& args [[buffer(0)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  const device T* a =
      reinterpret_cast<const device T*>(args.a.addr + args.a.offset);
  const device T* b =
      reinterpret_cast<const device T*>(args.b.addr + args.b.offset);
  device U* c = reinterpret_cast<device U*>(args.c.addr + args.c.offset);
  const int64_t size = args.size;

  int64_t offset = N * (index.x + grid_dim.x * int64_t(index.y));
  if (N > 1 && offset + N > size) {
    for (int i = 0; offset + i < size; ++i) {
      c[offset + i] = Op()(a[offset + i], b[offset + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[offset + i] = Op()(a[offset + i], b[offset + i]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void unary_v_ab(
    constant const UnaryVArgs& args [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  const device T* a =
      reinterpret_cast<const device T*>(args.a.addr + args.a.offset);
  device U* c = reinterpret_cast<device U*>(args.c.addr + args.c.offset);
  const uint size = static_cast<uint>(args.size);

  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      c[index + i] = Op()(a[index + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[index + i] = Op()(a[index + i]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void unary_v2_ab(
    constant const UnaryVArgs& args [[buffer(0)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  const device T* a =
      reinterpret_cast<const device T*>(args.a.addr + args.a.offset);
  device U* c = reinterpret_cast<device U*>(args.c.addr + args.c.offset);
  const int64_t size = args.size;

  int64_t offset = N * (index.x + grid_dim.x * int64_t(index.y));
  if (N > 1 && offset + N > size) {
    for (int i = 0; offset + i < size; ++i) {
      c[offset + i] = Op()(a[offset + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[offset + i] = Op()(a[offset + i]);
    }
  }
}

// Binary instantiations — Add, Multiply (vv = default N, vv2 = int64-size,
// vvn = explicit N from WorkPerThread).
#define instantiate_binary_ab(op, tname, itype, otype)                   \
  instantiate_kernel("vv_ab_" #op #tname, binary_vv_ab, itype, otype, op, 1)  \
  instantiate_kernel("vvn_ab_" #op #tname, binary_vv_ab, itype, otype, op)   \
  instantiate_kernel("vv2_ab_" #op #tname, binary_vv2_ab, itype, otype, op)

#define instantiate_binary_ab_float(op)               \
  instantiate_binary_ab(op, float16, half, half)      \
  instantiate_binary_ab(op, float32, float, float)    \
  instantiate_binary_ab(op, bfloat16, bfloat16_t, bfloat16_t)

instantiate_binary_ab_float(Add)
instantiate_binary_ab_float(Multiply)

// Unary instantiations — Sigmoid (activation), Negative (sign
// flips), Abs, Exp (softmax prologue). Same-type (T == U). Each
// op picks up all three vector shapes (v, vn, v2).
#define instantiate_unary_ab(op, tname, type)                              \
  instantiate_kernel("v_ab_" #op #tname, unary_v_ab, type, type, op, 1)    \
  instantiate_kernel("vn_ab_" #op #tname, unary_v_ab, type, type, op)      \
  instantiate_kernel("v2_ab_" #op #tname, unary_v2_ab, type, type, op)

#define instantiate_unary_ab_float(op)                \
  instantiate_unary_ab(op, float16, half)             \
  instantiate_unary_ab(op, float32, float)            \
  instantiate_unary_ab(op, bfloat16, bfloat16_t)

instantiate_unary_ab_float(Sigmoid)
instantiate_unary_ab_float(Negative)
instantiate_unary_ab_float(Abs)
instantiate_unary_ab_float(Exp)
// clang-format on
