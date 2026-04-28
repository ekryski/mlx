// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention_nax, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 64, 32, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 64, 32,  64, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 64, 64, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 64, 64,  64, 4, 1, mname, mtype)

// BD=256 NAX instantiations are intentionally NOT generated here.
//
// `attention_nax<dtype, BQ=32, BK=16, BD=256, WM=4, WN=1>` triggers two
// compile-time failures inside the kernel body:
//   - `BQ >= kNWarps * kU` is 32 >= 64, false → static_assert fires
//   - downstream `TQ = BQ / (kNWarps * kU) = 0`, instantiating
//     `NAXTile<float, 0, …>` which has zero-length arrays in its layout
//
// The runtime SDPA dispatch already guards `q.shape(3) <= 128` for the NAX
// path (see scaled_dot_product_attention.cpp:178), so these instantiations
// were already dead code — never callable. The older Metal toolchain
// happened to accept them; the Xcode 26 / macOS 26 toolchain rejects the
// dead instantiations at compile time.
//
// BD=256 attention is still supported via the non-NAX Steel path
// (steel_attention.metal), which has the matching BD=256 instantiations.

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat);

instantiate_attn_mask_helper(float32, float);
// clang-format on
