// Copyright © 2026 Eric Kryski. TurboFlash fused single-pass SDPA — spec 041
// phase 1.1 follow-up.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void TurboFlashSDPA::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  const array& q = inputs[0];
  const array& k_packed = inputs[1];
  const array& k_norms = inputs[2];
  const array& k_codebook = inputs[3];
  const array& v_packed = inputs[4];
  const array& v_norms = inputs[5];
  const array& v_codebook = inputs[6];
  const array* sinks_arr = has_sinks_ ? &inputs[7] : nullptr;

  int total_q = q.shape(0);
  // token_count = N — number of K positions stored. Layout
  // `[B*nKV, N, packed]` so dim(1) is N.
  int N = k_packed.shape(1);

  // Function constants — match kernel-side IDs (60, 61).
  metal::MTLFCList func_consts = {
      {&has_sinks_, MTL::DataType::DataTypeBool, 60},
      {&do_causal_, MTL::DataType::DataTypeBool, 61},
  };

  // Kernel name: turbo_flash_sdpa_v_{kb}_{vb}_{dim}
  std::string kname = "turbo_flash_sdpa_v_" + std::to_string(key_bits_) + "_" +
      std::to_string(value_bits_) + "_" + std::to_string(dim_);
  std::string hash_name = kname + (has_sinks_ ? "_sinks" : "_nosinks") +
      (do_causal_ ? "_c" : "_nc");

  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k_packed, 1);
  compute_encoder.set_input_array(k_norms, 2);
  compute_encoder.set_input_array(k_codebook, 3);
  compute_encoder.set_input_array(v_packed, 4);
  compute_encoder.set_input_array(v_norms, 5);
  compute_encoder.set_input_array(v_codebook, 6);
  compute_encoder.set_output_array(out, 7);
  compute_encoder.set_bytes(N, 8);
  compute_encoder.set_bytes(repeat_count_, 9);
  if (has_sinks_) {
    compute_encoder.set_input_array(*sinks_arr, 10);
    // num_q_heads: total_q / B = nQ assuming B=1 (typical inference shape).
    // For B>1 callers must reshape inputs so each batch is its own block.
    int num_q_heads = total_q;
    compute_encoder.set_bytes(num_q_heads, 11);
  }
  if (do_causal_) {
    compute_encoder.set_bytes(window_size_, 12);
  }

  // Grid: (total_q, 1, 1). Threadgroup: (1024, 1, 1) = 32 simdgroups.
  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(total_q, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

bool TurboFlashSDPA::is_equivalent(const Primitive& other) const {
  const TurboFlashSDPA& o = static_cast<const TurboFlashSDPA&>(other);
  return key_bits_ == o.key_bits_ && value_bits_ == o.value_bits_ &&
      dim_ == o.dim_ && repeat_count_ == o.repeat_count_ &&
      has_sinks_ == o.has_sinks_ && do_causal_ == o.do_causal_ &&
      window_size_ == o.window_size_;
}

std::vector<Shape> TurboFlashSDPA::output_shapes(
    const std::vector<array>& inputs) {
  const array& q = inputs[0];
  return {{q.shape(0), dim_}};
}

} // namespace mlx::core::fast
