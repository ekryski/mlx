// Copyright © 2026 Apple Inc.
#pragma once

#include <cstdlib>

namespace mlx::core::metal {

// Shared gate for Argument-Buffer-backed primitive paths.
//
// Semantics:
//   MLX_METAL_AB=1              → AB paths active (unchanged from the
//                                 pre-existing 8 shipped primitives).
//   MLX_METAL_ICB=1             → AB paths active. ICB *implies* AB.
//                                 Rationale: ICB replay without AB is
//                                 numerically broken past the recorded
//                                 T_k (shape-sensitivity diagnostic at
//                                 mlx 6f097aa6). The "ICB without AB"
//                                 config is unreachable by design.
//   (neither set)               → legacy / alpha behavior.
//   MLX_METAL_AB=0 MLX_METAL_ICB=1 → still AB-on (ICB implies AB wins).
//
// Every per-primitive `*_ab_enabled()` function (rms, rope, binary,
// unary, affine_qmv, affine_gather_qmv, gather_front, compiled JIT,
// and the new unified SDPA) delegates to this single helper so the
// composition is defined in exactly one place.
inline bool ab_enabled() {
  static const bool v = []() {
    const char* ab = std::getenv("MLX_METAL_AB");
    const char* icb = std::getenv("MLX_METAL_ICB");
    bool ab_on = (ab != nullptr && ab[0] == '1');
    bool icb_on = (icb != nullptr && icb[0] == '1');
    return ab_on || icb_on;
  }();
  return v;
}

} // namespace mlx::core::metal
