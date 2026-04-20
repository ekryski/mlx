// Copyright © 2026 Apple Inc.
//
// Gather/Embedding fast path — argument-buffer variant. Covers the
// decode-time gather_front dispatch: one index lookup per step,
// row-contiguous source, single-index-array. Called from
// `Gather::eval_gpu` for embedding lookups on every model.
//
// Low dispatch count per step (1), so the AB migration is mostly
// a pattern-consistency win rather than a perf driver. Included so
// every decode-path primitive is on the same binding model before
// SwitchLinear / elementwise / SDPA land.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/indexing/indexing.h"

// Byte layout — must match the ArgumentBuffer slot ordering
// declared in `Gather::eval_gpu`'s AB branch.
//
// NOTE: as of the persistent-AB refactor 2026-04-20, `indices` is
// NOT a slot in the AB. It's bound as a direct kernel buffer at
// slot 1 so the decode-loop orchestrator can override it race-free
// via the tag-binding path.
//
//   [ 0..15] src     : BufferPtrOffset  (device T*)
//   [16..31] out     : BufferPtrOffset  (device T*)
//   [32..39] stride  : int64
//   [40..43] size    : int
//   [44..47] _pad    : uint32  (round to 16-byte multiple)
struct BufferPtrOffset {
  uint64_t addr;
  uint64_t offset;
};

struct GatherFrontArgs {
  BufferPtrOffset src;
  BufferPtrOffset out;
  int64_t stride;
  int size;
  uint _pad;
};

template <typename T, typename IdxT, typename LocT, int N>
[[kernel]] void gather_front_ab(
    constant const GatherFrontArgs& args [[buffer(0)]],
    const device IdxT* indices [[buffer(1)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  const device T* src =
      reinterpret_cast<const device T*>(args.src.addr + args.src.offset);
  device T* out =
      reinterpret_cast<device T*>(args.out.addr + args.out.offset);

  auto idx = offset_neg_idx(indices[index.y], args.size);
  LocT src_idx = static_cast<LocT>(args.stride) * idx;
  LocT out_idx = static_cast<LocT>(args.stride) * index.y;

  int s_idx = N * index.x;
  for (int i = 0; i < N && s_idx < args.stride; ++i, ++s_idx) {
    out[out_idx + s_idx] = src[src_idx + s_idx];
  }
}

#define instantiate_gather_front_ab(tname, ttype, iname, itype, lname, ltype, N)      \
  instantiate_kernel(                                                                 \
      "gather_front_ab_" #tname "_" #iname "_" #lname "_" #N,                         \
      gather_front_ab, ttype, itype, ltype, N)

#define instantiate_gather_front_ab_idx(tname, ttype, iname, itype)           \
  instantiate_gather_front_ab(tname, ttype, iname, itype, int, int, 1)        \
  instantiate_gather_front_ab(tname, ttype, iname, itype, int, int, 2)        \
  instantiate_gather_front_ab(tname, ttype, iname, itype, int64_t, int64_t, 1)\
  instantiate_gather_front_ab(tname, ttype, iname, itype, int64_t, int64_t, 2)

#define instantiate_gather_front_ab_t(tname, ttype)              \
  instantiate_gather_front_ab_idx(tname, ttype, uint32, uint32_t)\
  instantiate_gather_front_ab_idx(tname, ttype, int32, int32_t)

instantiate_gather_front_ab_t(bool_, bool)
instantiate_gather_front_ab_t(uint8, uint8_t)
instantiate_gather_front_ab_t(uint16, uint16_t)
instantiate_gather_front_ab_t(uint32, uint32_t)
instantiate_gather_front_ab_t(uint64, uint64_t)
instantiate_gather_front_ab_t(int8, int8_t)
instantiate_gather_front_ab_t(int16, int16_t)
instantiate_gather_front_ab_t(int32, int32_t)
instantiate_gather_front_ab_t(int64, int64_t)
instantiate_gather_front_ab_t(float16, half)
instantiate_gather_front_ab_t(bfloat16, bfloat16_t)
instantiate_gather_front_ab_t(float32, float)
instantiate_gather_front_ab_t(complex64, complex64_t)
// clang-format on
