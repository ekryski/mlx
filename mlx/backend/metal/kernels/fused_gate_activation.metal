// Copyright © 2026 Eric Kryski.

// Fused dense gate-activation kernel.
// Input:  gateUp of shape [rows, 2*hidden_dims]
// Output: out    of shape [rows, hidden_dims]
// Computes: out[r, i] = activation(gateUp[r, i]) * gateUp[r, hidden_dims + i]
//           (for silu and gelu_approx — two-arg activations compose both halves)
//
// Replaces the Split + activation + Multiply chain (4 Metal dispatches) with a
// single dispatch. Decode-only fast path — at T=1 the dispatch overhead is
// the dominant cost on these element-wise ops.

#include <metal_common>
#include <metal_simdgroup>

#include "utils.h"

using namespace metal;

// ─── Activation functions ───────────────────────────────────────────────────

inline float silu_act(float x) {
  return x / (1.0f + metal::precise::exp(-x));
}

inline float gelu_approx_act(float x) {
  // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  constexpr float kSqrt2OverPi = 0.7978845608f;
  float x3 = x * x * x;
  float inner = kSqrt2OverPi * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + metal::precise::tanh(inner));
}

// activation_type: 0 = silu, 1 = gelu_approx
template <typename T, int activation_type>
[[kernel]] void fused_gate_activation(
    const device T* gateUp [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant uint& hidden_dims [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  const device T* gu_row = gateUp + gid * size_t(2 * hidden_dims);
  device T* o_row = out + gid * size_t(hidden_dims);

  for (uint i = lid; i < hidden_dims; i += tg_size) {
    float g = static_cast<float>(gu_row[i]);
    float u = static_cast<float>(gu_row[hidden_dims + i]);
    float result;
    if (activation_type == 0) {
      result = silu_act(g) * u;
    } else {
      result = gelu_approx_act(g) * u;
    }
    o_row[i] = static_cast<T>(result);
  }
}

#define instantiate_fused_gate_activation(name, type, act_name, act_id)      \
  template [[host_name("fused_gate_activation_" #name "_act" #act_name)]]    \
  [[kernel]] void fused_gate_activation<type, act_id>(                       \
      const device type*, device type*, constant uint&, uint, uint, uint);

instantiate_fused_gate_activation(float32, float, 0, 0)
instantiate_fused_gate_activation(float32, float, 1, 1)
instantiate_fused_gate_activation(float16, half, 0, 0)
instantiate_fused_gate_activation(float16, half, 1, 1)
instantiate_fused_gate_activation(bfloat16, bfloat16_t, 0, 0)
instantiate_fused_gate_activation(bfloat16, bfloat16_t, 1, 1)
