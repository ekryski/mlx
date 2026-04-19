// Copyright © 2024 Apple Inc.
#include <cstdlib>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/ab_gate.h"
#include "mlx/backend/metal/argument_buffer.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/persistent_ab.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

// Debug-only override for A/B comparison in the regression harness.
// Forces the legacy `sdpa_vector` / `sdpa_vector_2pass` code paths
// even when the shared AB gate is on. NOT documented as a user knob —
// only the Phase 0/1 test binary flips this.
bool sdpa_force_legacy() {
  const char* e = std::getenv("MLX_SDPA_FORCE_LEGACY");
  return e != nullptr && e[0] == '1';
}

// Debug-only override for Phase 2 bisection: when set, the unified
// path uses the Phase 1 non-AB kernel body (individual setBytes /
// setBuffer calls) instead of the Phase 2 argument-buffer bind.
// Harness uses this to diff AB-wrap vs non-AB unified output. NOT a
// user knob.
bool sdpa_force_no_ab() {
  const char* e = std::getenv("MLX_SDPA_NO_AB");
  return e != nullptr && e[0] == '1';
}

void sdpa_full_self_attention_nax(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  using namespace mlx::steel;

  int wm = 4;
  int wn = 1;

  int bd = q.shape(-1);
  int bq = 64;
  int bk = 32;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2);
  int kL = k.shape(2);

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value();
  const bool do_causal = do_causal_;
  const bool has_sinks = sinks.has_value();

  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
      {&has_sinks, MTL::DataType::DataTypeBool, 302}};

  std::string base_name;
  concatenate(
      base_name,
      "steel_attention_",
      type_to_name(q),
      "_bq",
      bq,
      "_bk",
      bk,
      "_bd",
      bd,
      "_wm",
      wm,
      "_wn",
      wn,
      "_mask",
      type_to_name(has_mask ? *mask : q));

  std::string hash_name;
  concatenate(
      hash_name,
      base_name,
      "_align_Q_",
      (align_Q ? 't' : 'n'),
      "_align_K_",
      (align_K ? 't' : 'n'),
      "_has_mask_",
      (has_mask ? 't' : 'n'),
      "_do_causal_",
      (do_causal ? 't' : 'n'),
      "_has_sinks_",
      (has_sinks ? 't' : 'n'));

  auto& compute_encoder = metal::get_command_encoder(s);

  auto kernel = get_steel_attention_nax_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      q,
      bq,
      bk,
      bd,
      wm,
      wn,
      (has_mask ? *mask : q));

  compute_encoder.set_compute_pipeline_state(kernel);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int NQ = */ NQ,
      /* int NK = */ NK,

      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,

      /* int qL_rem = */ (qL - NQ_aligned * bq),
      /* int kL_rem = */ (kL - NK_aligned * bk),
      /* int qL_off = */ (kL - qL),

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_bytes(params, 4);

  if (has_mask) {
    auto& m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 7);
  }

  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_full_self_attention_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  if (metal::is_nax_available() && q.shape(3) != 80 &&
      q.shape(3) <= 128 &&  // NAX BD=256 has zero-length array bug in NAXTile
      (env::enable_tf32() || q.dtype() != float32)) {
    return sdpa_full_self_attention_nax(
        /* const Stream& s = */ s,
        /* metal::Device& d = */ d,
        /* const array& q = */ q,
        /* const array& k = */ k,
        /* const array& v = */ v,
        /* const float scale = */ scale,
        /* array& o = */ o,
        /* bool do_causal_ = */ do_causal_,
        /* const std::optional<array>& mask = */ mask,
        /* const std::optional<array>& sinks = */ sinks);
  }

  using namespace mlx::steel;

  int wm = 4;
  int wn = 1;

  int bd = q.shape(-1);
  int bq = 32;
  int bk = bd < 128 ? 32 : 16;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2);
  int kL = k.shape(2);

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value();
  const bool do_causal = do_causal_;
  const bool has_sinks = sinks.has_value();

  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
      {&has_sinks, MTL::DataType::DataTypeBool, 302}};

  std::string base_name;
  concatenate(
      base_name,
      "steel_attention_",
      type_to_name(q),
      "_bq",
      bq,
      "_bk",
      bk,
      "_bd",
      bd,
      "_wm",
      wm,
      "_wn",
      wn,
      "_mask",
      type_to_name(has_mask ? *mask : q));

  std::string hash_name;
  concatenate(
      hash_name,
      base_name,
      "_align_Q_",
      (align_Q ? 't' : 'n'),
      "_align_K_",
      (align_K ? 't' : 'n'),
      "_has_mask_",
      (has_mask ? 't' : 'n'),
      "_do_causal_",
      (do_causal ? 't' : 'n'),
      "_has_sinks_",
      (has_sinks ? 't' : 'n'));

  auto& compute_encoder = metal::get_command_encoder(s);

  auto kernel = get_steel_attention_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      q,
      bq,
      bk,
      bd,
      wm,
      wn,
      (has_mask ? *mask : q));

  compute_encoder.set_compute_pipeline_state(kernel);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int NQ = */ NQ,
      /* int NK = */ NK,

      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,

      /* int qL_rem = */ (qL - NQ_aligned * bq),
      /* int kL_rem = */ (kL - NK_aligned * bk),
      /* int qL_off = */ (kL - qL),

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_bytes(params, 4);

  if (has_mask) {
    auto& m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 7);
  }

  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks" : "_nosinks";

  // Get the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(gqa_factor, 4);
  compute_encoder.set_bytes(N, 5);
  compute_encoder.set_bytes(k_head_stride, 6);
  compute_encoder.set_bytes(k_seq_stride, 7);
  compute_encoder.set_bytes(v_head_stride, 8);
  compute_encoder.set_bytes(v_seq_stride, 9);

  compute_encoder.set_bytes(scale, 10);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 11 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 13);
    compute_encoder.set_bytes(q_seq_stride, 14);
    compute_encoder.set_bytes(head_stride, 15);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 16);
    compute_encoder.set_bytes(q.shape(1), 17);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector_2pass(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_2pass_1_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int n_simds = gqa_factor * q.shape(2);

  char devc = d.get_architecture().back();
  int N = k.shape(2);
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) {
        blocks = 512;
      } else if (N >= 65536) {
        blocks = 1024;
      }
    }
  } else {
    if (n_simds >= 4) {
      blocks = 64;
    } else {
      blocks = 32;
    }
  }
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];
  MTL::Size group_dims(32, gqa_factor, q.shape(2));
  MTL::Size grid_dims(k.shape(1), q.shape(0), blocks);

  // Allocate the intermediates
  Shape intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(out.shape().back());
  array intermediate(intermediate_shape, q.dtype(), nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.add_temporary(intermediate);
  compute_encoder.add_temporary(sums);
  compute_encoder.add_temporary(maxs);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
      {&blocks, MTL::DataType::DataTypeInt, 26},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks_" : "_nosinks_";
  hash_name += std::to_string(blocks);

  // Get the kernel
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  check_kernel_threadgroup_size(kernel, group_dims, hash_name);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(intermediate, 3);
  compute_encoder.set_output_array(sums, 4);
  compute_encoder.set_output_array(maxs, 5);
  compute_encoder.set_bytes(N, 7);
  compute_encoder.set_bytes(k_head_stride, 8);
  compute_encoder.set_bytes(k_seq_stride, 9);
  compute_encoder.set_bytes(v_head_stride, 10);
  compute_encoder.set_bytes(v_seq_stride, 11);
  compute_encoder.set_bytes(scale, 12);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 13 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 15);
    compute_encoder.set_bytes(q_seq_stride, 16);
    compute_encoder.set_bytes(head_stride, 17);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 18);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Final pass
  kname.clear();
  kname = "sdpa_vector_2pass_2_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Get the kernel
  kernel = d.get_kernel(kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(blocks, 4);

  // Launch
  group_dims = MTL::Size(1024, 1, 1);
  grid_dims = MTL::Size(q.shape(0) * q.shape(1), q.shape(2), 1);
  check_kernel_threadgroup_size(kernel, group_dims, kname);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// Phase 1 unified vector-SDPA CPU dispatch. Replaces the
// `sdpa_vector` / `sdpa_vector_2pass` branch when ab_enabled(). Same
// kernel argument layout as legacy `sdpa_vector` plus one extra
// runtime `blocks` scalar at buffer(11). Function-constant slots are
// disjoint from legacy (30..35 vs 20..25) so PSOs live in separate
// cache entries.
void sdpa_vector_unified(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks,
    const std::shared_ptr<metal::PersistentAb>& ab_handle = {}) {
  // Phase 2: default path is the AB-wrapped kernel. The non-AB Phase
  // 1 body is kept only as an internal bisection escape, reachable
  // via MLX_SDPA_NO_AB=1 (debug-only, not user-facing).
  const bool use_ab = !sdpa_force_no_ab();

  std::string kname;
  kname.reserve(64);
  kname += use_ab ? "sdpa_unified_vector_ab_" : "sdpa_unified_vector_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  // Phase 1: single-threadgroup-per-(batch*head, q_seq) layout.
  // `blocks` is reserved for Phase 2+; always 1 in this phase.
  int blocks = 1;

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 30},
      {&query_transposed, MTL::DataType::DataTypeBool, 31},
      {&do_causal, MTL::DataType::DataTypeBool, 32},
      {&bool_mask, MTL::DataType::DataTypeBool, 33},
      {&float_mask, MTL::DataType::DataTypeBool, 34},
      {&has_sinks, MTL::DataType::DataTypeBool, 35},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks" : "_nosinks";

  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  if (use_ab) {
    // Pack every kernel argument into a single shared-storage buffer
    // bound at slot 0. Layout must match `SdpaUnifiedArgs` in
    // `kernels/sdpa_unified.h`.
    //
    // When a caller-owned PersistentAb handle is supplied, rewrite
    // its contents in place and bind — no add_temporary_object,
    // since the caller owns the handle lifetime. Otherwise allocate
    // a fresh transient AB (same as pre-Option-A behavior).
    using Slot = metal::ArgumentBuffer::Slot;
    const std::vector<Slot> slot_layout{
        {Slot::Kind::BufferPtrOffset, 0, "queries"},
        {Slot::Kind::BufferPtrOffset, 0, "keys"},
        {Slot::Kind::BufferPtrOffset, 0, "values"},
        {Slot::Kind::BufferPtrOffset, 0, "out"},
        {Slot::Kind::BufferPtrOffset, 0, "mask"},
        {Slot::Kind::BufferPtrOffset, 0, "sinks"},
        {Slot::Kind::Scalar64, 0, "k_head_stride"},
        {Slot::Kind::Scalar64, 0, "k_seq_stride"},
        {Slot::Kind::Scalar64, 0, "v_head_stride"},
        {Slot::Kind::Scalar64, 0, "v_seq_stride"},
        {Slot::Kind::Float32, 0, "scale"},
        {Slot::Kind::Scalar32, 0, "gqa_factor"},
        {Slot::Kind::Scalar32, 0, "N"},
        {Slot::Kind::Scalar32, 0, "blocks"},
        {Slot::Kind::Scalar32, 0, "mask_kv_seq_stride"},
        {Slot::Kind::Scalar32, 0, "mask_q_seq_stride"},
        {Slot::Kind::Scalar32, 0, "mask_head_stride"},
        {Slot::Kind::Scalar32, 0, "num_q_heads"},
    };

    // Shared setter — called on either the persistent handle or a
    // fresh transient AB. Takes a generic callable for the setter
    // methods so we don't need to parameterize over types.
    auto populate = [&](auto& target) {
      target.set_buffer_ptr(
          0, static_cast<const MTL::Buffer*>(q.buffer().ptr()), q.offset());
      target.set_buffer_ptr(
          1, static_cast<const MTL::Buffer*>(k.buffer().ptr()), k.offset());
      target.set_buffer_ptr(
          2, static_cast<const MTL::Buffer*>(v.buffer().ptr()), v.offset());
      target.set_buffer_ptr(
          3, static_cast<const MTL::Buffer*>(out.buffer().ptr()), out.offset());
      if (has_mask) {
        const auto& m = *mask;
        target.set_buffer_ptr(
            4, static_cast<const MTL::Buffer*>(m.buffer().ptr()), m.offset());
        int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
        int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
        int32_t head_stride =
            m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
        target.set_scalar32(14, static_cast<uint32_t>(kv_seq_stride));
        target.set_scalar32(15, static_cast<uint32_t>(q_seq_stride));
        target.set_scalar32(16, static_cast<uint32_t>(head_stride));
      }
      if (has_sinks) {
        const auto& sn = *sinks;
        target.set_buffer_ptr(
            5, static_cast<const MTL::Buffer*>(sn.buffer().ptr()), sn.offset());
        target.set_scalar32(17, static_cast<uint32_t>(q.shape(1)));
      }
      target.set_scalar64(6, static_cast<uint64_t>(k_head_stride));
      target.set_scalar64(7, static_cast<uint64_t>(k_seq_stride));
      target.set_scalar64(8, static_cast<uint64_t>(v_head_stride));
      target.set_scalar64(9, static_cast<uint64_t>(v_seq_stride));
      target.set_float32(10, scale);
      target.set_scalar32(11, static_cast<uint32_t>(gqa_factor));
      target.set_scalar32(12, static_cast<uint32_t>(N));
      target.set_scalar32(13, static_cast<uint32_t>(blocks));
    };

    MTL::Buffer* ab_mtl = nullptr;
    if (ab_handle) {
      if (ab_handle->layout().size() != slot_layout.size()) {
        throw std::runtime_error(
            "[sdpa_vector_unified] persistent AB handle has wrong slot "
            "count (expected 18, got " +
            std::to_string(ab_handle->layout().size()) + ")");
      }
      populate(*ab_handle);
      ab_mtl = ab_handle->mtl_buffer();
    } else {
      auto ab = std::make_shared<metal::ArgumentBuffer>(d, slot_layout);
      populate(*ab);
      ab_mtl = ab->mtl_buffer();
      compute_encoder.add_temporary_object(
          std::static_pointer_cast<void>(ab));
    }

    // Fence tracking for resources the kernel reads/writes through
    // the AB (no binding — AB carries the addresses).
    compute_encoder.register_input_array(q);
    compute_encoder.register_input_array(k);
    compute_encoder.register_input_array(v);
    if (has_mask) compute_encoder.register_input_array(*mask);
    if (has_sinks) compute_encoder.register_input_array(*sinks);
    compute_encoder.register_output_array(out);

    compute_encoder.set_buffer(ab_mtl, 0);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  } else {
    // Phase 1 non-AB body — preserved as an internal debug escape
    // (MLX_SDPA_NO_AB=1). Kept for post-ship bisection only.
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(gqa_factor, 4);
    compute_encoder.set_bytes(N, 5);
    compute_encoder.set_bytes(k_head_stride, 6);
    compute_encoder.set_bytes(k_seq_stride, 7);
    compute_encoder.set_bytes(v_head_stride, 8);
    compute_encoder.set_bytes(v_seq_stride, 9);
    compute_encoder.set_bytes(scale, 10);
    compute_encoder.set_bytes(blocks, 11);
    if (has_mask) {
      auto& m = *mask;
      compute_encoder.set_input_array(m, 12 + float_mask);
      int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
      int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
      int32_t head_stride =
          m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
      compute_encoder.set_bytes(kv_seq_stride, 14);
      compute_encoder.set_bytes(q_seq_stride, 15);
      compute_encoder.set_bytes(head_stride, 16);
    }
    if (has_sinks) {
      compute_encoder.set_input_array(*sinks, 17);
      compute_encoder.set_bytes(q.shape(1), 18);
    }
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s,
    int window_size) {
  if (is_training) {
    // It's faster for training on Metal to use the unfused SDPA for both
    // forward and backward.
    return true;
  }
  if (output_logsumexp) {
    return true;
  }
  if (s.device == Device::cpu) {
    return true;
  }
  // Sliding-window masks are synthesized in the composed fallback path
  // (arange + compare on GPU). None of the current Steel / sdpa_vector /
  // NAX kernels honor an additional window bound — they treat the causal
  // mask as a full lower triangle. Route windowed cases through fallback.
  if (window_size > 0) {
    return true;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);
  const int num_query_heads = q.shape(1);
  const int num_kv_heads = k.shape(1);
  const int gqa_factor = num_query_heads / num_kv_heads;

  // BD=256 Steel kernels only exist for float16/bfloat16 — float32 exceeds
  // the 32KB threadgroup memory limit (41KB for BD=256 at 4B/elem).
  const bool is_half = q.dtype() == float16 || q.dtype() == bfloat16;

  // BD=512 vector kernel is instantiated for f16/bf16 — required for Gemma 4
  // full-attention layers (head_dim=512) during decode. Not enabled for Steel
  // full attention (prefill) due to register pressure at BD=512.
  const bool sdpa_vector_supported_head_dim =
      query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128 ||
       query_head_dim == 256 ||
       (query_head_dim == 512 && is_half));

  // Steel full attention avoids materializing L×L attention score matrices.
  // BD≤128: works for all dtypes.
  // BD=256: float16/bfloat16 only (threadgroup memory constraint).
  const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128 ||
       (query_head_dim == 256 && is_half));

  const bool sdpa_full_supported_mask = !has_mask || has_arr_mask ||
      (query_sequence_length <= key_sequence_length && do_causal);

  const bool supports_sdpa_full = query_sequence_length > 8 &&
      sdpa_full_supported_mask && sdpa_full_supported_head_dim;

  const bool supports_sdpa_vector = (query_sequence_length <= 8) &&
      (query_sequence_length <= key_sequence_length) &&
      sdpa_vector_supported_head_dim &&
      (query_sequence_length * gqa_factor) <= 32;

  return !(supports_sdpa_full || supports_sdpa_vector);
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return true;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_pre = inputs[0];
  auto& k_pre = inputs[1];
  auto& v_pre = inputs[2];
  auto& o = outputs[0];

  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  copies.reserve(inputs.size());
  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  std::optional<array> sinks = std::nullopt;
  if (has_sinks_) {
    sinks = copy_unless(is_matrix_contiguous, inputs.back());
  }
  bool has_arr_mask = inputs.size() > (3 + has_sinks_);

  // We are in vector mode ie single query
  if (q_pre.shape(2) <= 8) {
    auto q_copy_unless = [](const array& arr) {
      if (arr.flags().row_contiguous) {
        return true;
      }
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (shape[0] == 1 || shape[1] == 1) {
        // If either the batch or head dimension is a singleton, the other can
        // be transposed with the sequence dimension
        auto bidx = shape[0] == 1 ? 1 : 0;
        return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
            (strides[bidx] == shape[3]);
      }
      return false;
    };

    auto kv_copy_unless = [](const array& arr) {
      // keys and values should be copied if:
      // - the last dimension is not contiguous
      // - the batch and head dim are not contiguous
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (strides.back() != 1) {
        return false;
      }
      if (shape[0] == 1 || shape[1] == 1) {
        return true;
      }
      return (strides[0] == strides[1] * shape[1]);
    };

    bool q_copied = !q_copy_unless(q_pre);
    array q = (q_copied) ? contiguous_copy_gpu(q_pre, s) : q_pre;
    const auto& k = copy_unless(kv_copy_unless, k_pre);
    const auto& v = copy_unless(kv_copy_unless, v_pre);

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      if (q_copied) {
        copies.push_back(q);
      }
      o.set_data(allocator::malloc(o.nbytes()));
    }

    auto mask_copy_unless = [&q](const array& arr) {
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      return arr.flags().row_contiguous || q.shape(0) == 1 || q.shape(1) == 1 ||
          (strides[0] == strides[1] * shape[1]);
    };

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(mask_copy_unless, inputs[3])}
        : std::nullopt;

    // We route to the 2 pass fused attention if
    // - The device is large and the sequence length long
    // - The sequence length is even longer and we have gqa
    bool do_causal = do_causal_ && q.shape(2) > 1;
    char devc = d.get_architecture().back();
    // Phase 1 of Option C: when AB gate is on (MLX_METAL_AB or
    // MLX_METAL_ICB), route to the unified vector kernel for all T_k.
    // This removes the topology flip that breaks ICB replay past the
    // recorded T_k. MLX_SDPA_FORCE_LEGACY=1 is a debug-only override
    // used by the regression harness for within-binary A/B comparison.
    if (metal::ab_enabled() && !sdpa_force_legacy()) {
      sdpa_vector_unified(
          s, d, q, k, v, o, scale_, do_causal, mask, sinks, ab_handle());
    } else if (
        ((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
        (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
      sdpa_vector_2pass(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
    } else {
      sdpa_vector(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
    }
  }

  // Full attention mode
  else {
    const auto& q = copy_unless(is_matrix_contiguous, q_pre);
    const auto& k = copy_unless(is_matrix_contiguous, k_pre);
    const auto& v = copy_unless(is_matrix_contiguous, v_pre);

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;

    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };

    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(is_matrix_contiguous, inputs[3])}
        : std::nullopt;

    sdpa_full_self_attention_metal(
        s, d, q, k, v, scale_, o, do_causal_, mask, sinks);
  }

  metal::get_command_encoder(s).add_temporaries(std::move(copies));
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  return true;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("NYI");
}

} // namespace mlx::core::fast
