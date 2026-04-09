// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype)

// BD=256: BQ=16, WM=2 for smaller output tile + better occupancy, float16/bfloat16 only
#define instantiate_attn_shapes_helper_bd256(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 16, 16, 256, 2, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

#define instantiate_attn_mask_helper_bd256(iname, itype) \
    instantiate_attn_shapes_helper_bd256(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper_bd256(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper_bd256(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat16_t);
instantiate_attn_mask_helper_bd256(bfloat16, bfloat16_t);

instantiate_attn_mask_helper(float32, float);
// BD=256 and BD=512 are NOT instantiated for float32 — threadgroup memory
// exceeds 32KB limit (41KB for BD=256, 49KB for BD=512 at 4 bytes/element).
// Only float16/bfloat16 (2 bytes/element) fit. Dispatch gates on dtype.

// BD=512: BQ=8, BK=8, WM=1, WN=1 — minimal tile for large head dimensions.
// High register pressure (512 regs/thread for Otile) limits occupancy to ~6%
// on M1 Max, but avoids materializing L×L attention score matrices during prefill.
// Potentially viable on M5 Max with larger register files. Gated behind runtime check
// in scaled_dot_product_attention.cpp.
#define instantiate_attn_shapes_helper_bd512(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 8, 8, 512, 1, 1, mname, mtype)

#define instantiate_attn_mask_helper_bd512(iname, itype) \
    instantiate_attn_shapes_helper_bd512(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper_bd512(iname, itype, bool_, bool)

instantiate_attn_mask_helper_bd512(float16, half);
instantiate_attn_mask_helper_bd512(bfloat16, bfloat16_t);
// clang-format on
