// Copyright © 2026 Apple Inc.
//
// RoPE — argument-buffer variants of the single-token (decode) kernels.
// `rope_single` and `rope_single_freqs` — both are the T==1,
// row-contiguous, one-offset fast paths.
// `rope` and `rope_freqs` (prefill variants) remain on the legacy path
// via rope.metal. See benchmarks/notes/ab-rmsnorm-pilot-2026-04-17.md
// for the broader design pattern.

#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"

constant bool forward [[function_constant(1)]];
constant bool traditional [[function_constant(2)]];

// Shared byte layout for the single-token AB path — must match the
// ArgumentBuffer slot ordering declared by RoPE::eval_gpu for the AB
// single-token branch.
//
// NOTE: as of the persistent-AB refactor 2026-04-20, `offset` is
// NOT a slot in the AB. It's bound as a direct kernel buffer at
// slot 1 so the decode-loop orchestrator can override it race-free
// via the tag-binding path.
//
// Non-freqs (passes `base` inline):
//   [ 0..15] in     : BufferPtrOffset
//   [16..31] out    : BufferPtrOffset
//   [32..35] scale  : float
//   [36..39] _pad0  : uint32
//   [40..47] stride : int64
//   [48..51] base   : float
//
// Freqs (device-pointer to freqs table + freq_stride):
//   [ 0..15] in           : BufferPtrOffset
//   [16..31] out          : BufferPtrOffset
//   [32..35] scale        : float
//   [36..39] _pad0        : uint32
//   [40..47] stride       : int64
//   [48..63] freqs        : BufferPtrOffset
//   [64..71] freq_stride  : int64
struct BufferPtrOffset {
  uint64_t addr;
  uint64_t offset;
};

struct RopeSingleArgs {
  BufferPtrOffset in;
  BufferPtrOffset out;
  float scale;
  uint _pad0;
  int64_t stride;
  float base;
};

struct RopeSingleFreqsArgs {
  BufferPtrOffset in;
  BufferPtrOffset out;
  float scale;
  uint _pad0;
  int64_t stride;
  BufferPtrOffset freqs;
  int64_t freq_stride;
};

// Shared body — identical math to `rope_single_impl` in rope.metal.
template <typename T>
inline void rope_single_body(
    const device T* in,
    device T* out,
    const int offset_val,
    const float inv_freq,
    const float scale,
    const int64_t stride,
    uint2 pos,
    uint2 grid) {
  float L = scale * static_cast<float>(offset_val);

  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

  uint index_1, index_2;
  if (traditional) {
    index_1 = 2 * pos.x + pos.y * stride;
    index_2 = index_1 + 1;
  } else {
    index_1 = pos.x + pos.y * stride;
    index_2 = index_1 + grid.x;
  }

  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  float rx1;
  float rx2;
  if (forward) {
    rx1 = x1 * costheta - x2 * sintheta;
    rx2 = x1 * sintheta + x2 * costheta;
  } else {
    rx1 = x2 * sintheta + x1 * costheta;
    rx2 = x2 * costheta - x1 * sintheta;
  }
  out[index_1] = static_cast<T>(rx1);
  out[index_2] = static_cast<T>(rx2);
}

template <typename T>
[[kernel]] void rope_ab_single(
    constant const RopeSingleArgs& args [[buffer(0)]],
    const device int* offset_ptr [[buffer(1)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const device T* in =
      reinterpret_cast<const device T*>(args.in.addr + args.in.offset);
  device T* out =
      reinterpret_cast<device T*>(args.out.addr + args.out.offset);

  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * args.base);

  rope_single_body<T>(
      in, out, offset_ptr[0], inv_freq, args.scale, args.stride, pos, grid);
}

template <typename T>
[[kernel]] void rope_ab_single_freqs(
    constant const RopeSingleFreqsArgs& args [[buffer(0)]],
    const device int* offset_ptr [[buffer(1)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const device T* in =
      reinterpret_cast<const device T*>(args.in.addr + args.in.offset);
  device T* out =
      reinterpret_cast<device T*>(args.out.addr + args.out.offset);
  const device float* freqs =
      reinterpret_cast<const device float*>(args.freqs.addr + args.freqs.offset);

  float inv_freq = 1.0 / (freqs[args.freq_stride * pos.x]);

  rope_single_body<T>(
      in, out, offset_ptr[0], inv_freq, args.scale, args.stride, pos, grid);
}

// clang-format off
#define instantiate_rope_ab(name, type)                                  \
  instantiate_kernel("rope_ab_single_" #name, rope_ab_single, type)      \
  instantiate_kernel("rope_ab_single_freqs_" #name, rope_ab_single_freqs, type)

instantiate_rope_ab(float16, half)
instantiate_rope_ab(bfloat16, bfloat16_t)
instantiate_rope_ab(float32, float) // clang-format on
