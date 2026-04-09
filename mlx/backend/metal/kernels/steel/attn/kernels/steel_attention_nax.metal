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

// BD=256 only for half/bfloat16 — float32 exceeds 32KB threadgroup memory limit
#define instantiate_attn_shapes_helper_bd256(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 64, 16, 256, 4, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

#define instantiate_attn_mask_helper_bd256(iname, itype) \
    instantiate_attn_shapes_helper_bd256(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper_bd256(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper_bd256(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat);
instantiate_attn_mask_helper_bd256(bfloat16, bfloat);

instantiate_attn_mask_helper(float32, float);
// clang-format on
