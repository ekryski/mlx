// Copyright © 2023-2024 Apple Inc.
#include <functional>
#include <unordered_map>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

namespace detail {

// ─────────────────────────────────────────────────────────────────────
// PinSession — stable-address allocator reuse for decode-loop ICB
// ─────────────────────────────────────────────────────────────────────
//
// A PinSession captures the shared_ptr<array::Data> produced by every
// `array::set_data` call on the current thread during a "record"
// phase, and reuses those same Data instances during a subsequent
// "replay" phase. When a primitive's `eval_gpu` does
// `out.set_data(allocator::malloc(nbytes))`, the pin session (in
// replay phase) discards the freshly-allocated buffer and attaches
// `out` to the recorded Data — so `out.buffer().ptr()` returns the
// *same* MTLBuffer address across steps.
//
// This is the foundation for Option (b) of decode-loop ICB
// orchestration: the recorded ICB's every binding remains valid
// across replays because every allocation in the forward pass has a
// stable address. No per-binding override mechanism is needed.
//
// One session per thread (matches mlx's single-threaded-decode
// assumption). Sessions carry no cross-thread state.

struct PinSession {
  enum class Phase { None, Record, Replay };
  Phase phase{Phase::None};
  // Captured shared_ptrs — one per set_data call during Record.
  // During Replay, slot_idx walks this vector and each reused Data
  // is written back into the caller's `array_desc_->data`.
  std::vector<std::shared_ptr<array::Data>> slots;
  size_t slot_idx{0};
  // Diagnostic: total calls seen during the current phase. Helps
  // detect record/replay divergence (different graph topology).
  size_t calls{0};

  void begin_record() {
    phase = Phase::Record;
    slots.clear();
    slot_idx = 0;
    calls = 0;
  }
  size_t end_record() {
    phase = Phase::None;
    return slots.size();
  }
  void begin_replay() {
    phase = Phase::Replay;
    slot_idx = 0;
    calls = 0;
  }
  size_t end_replay() {
    size_t consumed = slot_idx;
    phase = Phase::None;
    return consumed;
  }

  // Called from array::set_data with the just-constructed Data.
  // In Record: capture.
  // In Replay: replace with the next recorded slot.
  // In None: no-op.
  void on_set_data(std::shared_ptr<array::Data>& data) {
    if (phase == Phase::Record) {
      slots.push_back(data);
      ++calls;
    } else if (phase == Phase::Replay) {
      ++calls;
      if (slot_idx < slots.size()) {
        // Swap: the freshly-allocated Data falls out of scope and
        // returns its buffer to the pool; the recorded Data takes
        // its place so the caller's array pins at the recorded
        // buffer address.
        data = slots[slot_idx++];
      }
      // Underflow (more set_data calls than recorded): fall through
      // with the fresh Data. Indicates record/replay divergence;
      // caller should detect via the count mismatch returned by
      // end_replay.
    }
  }
};

static thread_local PinSession* t_pin_session_ = nullptr;

MLX_API PinSession* current_pin_session() {
  return t_pin_session_;
}

MLX_API PinSession* begin_pin_record_session() {
  if (t_pin_session_) {
    throw std::logic_error(
        "[mlx::detail] pin session already active on this thread");
  }
  auto* sess = new PinSession();
  sess->begin_record();
  t_pin_session_ = sess;
  return sess;
}

MLX_API size_t end_pin_record_session() {
  if (!t_pin_session_) {
    throw std::logic_error(
        "[mlx::detail] end_pin_record_session without begin");
  }
  if (t_pin_session_->phase != PinSession::Phase::Record) {
    throw std::logic_error(
        "[mlx::detail] end_pin_record_session while not in Record phase");
  }
  size_t n = t_pin_session_->end_record();
  // Detach from TLS so the session can be re-attached later via
  // begin_pin_replay. The caller still owns the session handle.
  t_pin_session_ = nullptr;
  return n;
}

MLX_API void begin_pin_replay(PinSession* sess) {
  if (!sess) {
    throw std::invalid_argument(
        "[mlx::detail] begin_pin_replay: null session");
  }
  if (t_pin_session_) {
    throw std::logic_error(
        "[mlx::detail] begin_pin_replay while another session is active");
  }
  t_pin_session_ = sess;
  sess->begin_replay();
}

MLX_API size_t end_pin_replay() {
  if (!t_pin_session_) {
    throw std::logic_error(
        "[mlx::detail] end_pin_replay without begin");
  }
  if (t_pin_session_->phase != PinSession::Phase::Replay) {
    throw std::logic_error(
        "[mlx::detail] end_pin_replay while not in Replay phase");
  }
  size_t consumed = t_pin_session_->end_replay();
  t_pin_session_ = nullptr;
  return consumed;
}

MLX_API void free_pin_session(PinSession* sess) {
  // Caller is responsible for not freeing a session that's still
  // t_pin_session_ — we defensively null the TLS if it happens to
  // match, but deleting mid-session is a bug.
  if (t_pin_session_ == sess) {
    t_pin_session_ = nullptr;
  }
  delete sess;
}

MLX_API size_t pin_session_slot_count(PinSession* sess) {
  if (!sess) {
    return 0;
  }
  return sess->slots.size();
}

MLX_API size_t pin_session_slot_idx(PinSession* sess) {
  if (!sess) {
    return 0;
  }
  return sess->slot_idx;
}

} // namespace detail

array::array(const std::complex<float>& val, Dtype dtype /* = complex64 */)
    : array_desc_(std::make_shared<ArrayDesc>(Shape{}, dtype)) {
  auto cval = static_cast<complex64_t>(val);
  init(&cval);
}

array::array(
    Shape shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              std::move(shape),
              dtype,
              std::move(primitive),
              std::move(inputs))) {
  if (has_primitive() && this->primitive().stream().device == Device::gpu) {
    for (auto& in : this->inputs()) {
      if (in.dtype() == float64) {
        throw std::invalid_argument("float64 is not supported on the GPU");
      }
    }
    if (this->dtype() == float64) {
      throw std::invalid_argument("float64 is not supported on the GPU");
    }
  }
}

std::vector<array> array::make_arrays(
    std::vector<Shape> shapes,
    const std::vector<Dtype>& dtypes,
    const std::shared_ptr<Primitive>& primitive,
    const std::vector<array>& inputs) {
  std::vector<array> outputs;
  for (size_t i = 0; i < shapes.size(); ++i) {
    outputs.emplace_back(std::move(shapes[i]), dtypes[i], primitive, inputs);
  }
  // For each node in |outputs|, its siblings are the other nodes.
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto siblings = outputs;
    siblings.erase(siblings.begin() + i);
    outputs[i].set_siblings(std::move(siblings), i);
  }
  return outputs;
}

array array::unsafe_weak_copy(const array& other) {
  auto cpy = array(other.shape(), other.dtype(), nullptr, {});
  cpy.set_data(
      other.buffer(),
      other.data_size(),
      other.strides(),
      other.flags(),
      [](auto) {});
  cpy.array_desc_->offset = other.array_desc_->offset;
  return cpy;
}

array::array(std::initializer_list<float> data)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              Shape{static_cast<ShapeElem>(data.size())},
              float32)) {
  init(data.begin());
}

array::array(std::initializer_list<int> data, Dtype dtype)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              Shape{static_cast<ShapeElem>(data.size())},
              dtype)) {
  init(data.begin());
}

array::array(
    void* data,
    Shape shape,
    Dtype dtype,
    const std::function<void(void*)>& deleter)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  auto buffer = allocator::make_buffer(data, nbytes());
  if (buffer.ptr() == nullptr) {
    set_data(allocator::malloc(nbytes()));
    auto ptr = static_cast<char*>(data);
    std::copy(ptr, ptr + nbytes(), this->data<char>());
    deleter(data);
  } else {
    auto wrapped_deleter = [deleter](allocator::Buffer buffer) {
      auto ptr = buffer.raw_ptr();
      allocator::release(buffer);
      return deleter(ptr);
    };
    set_data(buffer, std::move(wrapped_deleter));
  }
}

/* Build an array from a shared buffer */
array::array(allocator::Buffer data, Shape shape, Dtype dtype, Deleter deleter)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  set_data(data, deleter);
}

void array::detach() {
  array_desc_->primitive = nullptr;
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->primitive = nullptr;
  }
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->inputs.clear();
    s.array_desc_->siblings.clear();
    s.array_desc_->position = 0;
  }
  array_desc_->inputs.clear();
  array_desc_->siblings.clear();
  array_desc_->position = 0;
}

bool array::is_available() const {
  if (status() == Status::available) {
    return true;
  } else if (
      status() == Status::evaluated &&
      (!event().valid() || event().is_signaled())) {
    detach_event();
    set_status(Status::available);
    return true;
  }
  return false;
}

void array::wait() {
  if (!is_available()) {
    if (event().valid()) {
      event().wait();
      detach_event();
    }
    set_status(Status::available);
  }
}

void array::eval() {
  // Ensure the array is ready to be read
  if (status() == Status::unscheduled) {
    mlx::core::eval({*this});
  } else {
    wait();
  }
}

bool array::is_tracer() const {
  return (array_desc_->is_tracer && detail::in_tracing()) ||
      detail::retain_graph();
}

void array::set_data(allocator::Buffer buffer, Deleter d) {
  auto data = std::make_shared<Data>(buffer, d);
  if (auto* sess = detail::current_pin_session()) {
    sess->on_set_data(data);
  }
  array_desc_->data = std::move(data);
  array_desc_->offset = 0;
  array_desc_->data_size = size();
  array_desc_->flags.contiguous = true;
  array_desc_->flags.row_contiguous = true;
  auto max_dim = std::max_element(shape().begin(), shape().end());
  array_desc_->flags.col_contiguous = size() <= 1 || size() == *max_dim;
}

void array::set_data(
    allocator::Buffer buffer,
    size_t data_size,
    Strides strides,
    Flags flags,
    Deleter d) {
  auto data = std::make_shared<Data>(buffer, d);
  if (auto* sess = detail::current_pin_session()) {
    sess->on_set_data(data);
  }
  array_desc_->data = std::move(data);
  array_desc_->offset = 0;
  array_desc_->data_size = data_size;
  array_desc_->strides = std::move(strides);
  array_desc_->flags = flags;
}

void array::copy_shared_buffer(
    const array& other,
    const Strides& strides,
    Flags flags,
    size_t data_size,
    int64_t offset /* = 0 */) {
  array_desc_->data = other.array_desc_->data;
  array_desc_->strides = strides;
  array_desc_->flags = flags;
  array_desc_->data_size = data_size;
  array_desc_->offset =
      sizeof(char) * itemsize() * offset + other.array_desc_->offset;
}

void array::copy_shared_buffer(const array& other) {
  copy_shared_buffer(other, other.strides(), other.flags(), other.data_size());
}

array::~array() {
  if (array_desc_ == nullptr) {
    return;
  }

  // Detached/detaching
  if (array_desc_->primitive == nullptr) {
    return;
  }

  // Break circular reference for non-detached arrays with siblings
  if (auto n = siblings().size(); n > 0) {
    bool do_detach = true;
    // If all siblings have siblings.size() references except
    // the one we are currently destroying (which has siblings.size() + 1)
    // then there are no more external references
    do_detach &= (array_desc_.use_count() == (n + 1));
    for (auto& s : siblings()) {
      do_detach &= (s.array_desc_.use_count() == n);
      if (!do_detach) {
        break;
      }
    }
    if (do_detach) {
      for (auto& s : siblings()) {
        for (auto& ss : s.siblings()) {
          // Set to null here to avoid descending into array destructor
          // for siblings
          ss.array_desc_ = nullptr;
        }
        s.array_desc_->siblings.clear();
      }
    }
  }
}

void array::ArrayDesc::init() {
  strides.resize(shape.size());
  size = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = size;
    size *= shape[i];
  }
  for (const auto& in : inputs) {
    is_tracer |= in.is_tracer();
  }
}

array::ArrayDesc::ArrayDesc(Shape shape, Dtype dtype)
    : shape(std::move(shape)), dtype(dtype), status(Status::available) {
  init();
}

array::ArrayDesc::ArrayDesc(
    Shape shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : shape(std::move(shape)),
      dtype(dtype),
      primitive(std::move(primitive)),
      status(Status::unscheduled),
      inputs(std::move(inputs)) {
  init();
}

array::ArrayDesc::~ArrayDesc() {
  // When an array description is destroyed it will delete a bunch of arrays
  // that may also destroy their corresponding descriptions and so on and so
  // forth.
  //
  // This calls recursively the destructor and can result in stack overflow, we
  // instead put them in a vector and destroy them one at a time resulting in a
  // max stack depth of 2.
  if (inputs.empty()) {
    return;
  }

  std::vector<std::shared_ptr<ArrayDesc>> for_deletion;

  auto append_deletable_inputs = [&for_deletion](ArrayDesc& ad) {
    std::unordered_map<std::uintptr_t, array> input_map;
    for (array& a : ad.inputs) {
      if (a.array_desc_) {
        input_map.insert({a.id(), a});
        for (auto& s : a.siblings()) {
          input_map.insert({s.id(), s});
        }
      }
    }
    ad.inputs.clear();
    for (auto& [_, a] : input_map) {
      bool is_deletable =
          (a.array_desc_.use_count() <= a.siblings().size() + 1);
      // An array with siblings is deletable only if all of its siblings
      // are deletable
      for (auto& s : a.siblings()) {
        if (!is_deletable) {
          break;
        }
        int is_input = (input_map.find(s.id()) != input_map.end());
        is_deletable &=
            s.array_desc_.use_count() <= a.siblings().size() + is_input;
      }
      if (is_deletable) {
        for_deletion.push_back(std::move(a.array_desc_));
      }
    }
  };

  append_deletable_inputs(*this);

  while (!for_deletion.empty()) {
    // top is going to be deleted at the end of the block *after* the arrays
    // with inputs have been moved into the vector
    auto top = std::move(for_deletion.back());
    for_deletion.pop_back();
    append_deletable_inputs(*top);

    // Clear out possible siblings to break circular references
    for (auto& s : top->siblings) {
      // Set to null here to avoid descending into top-level
      // array destructor for siblings
      s.array_desc_ = nullptr;
    }
    top->siblings.clear();
  }
}

array::ArrayIterator::ArrayIterator(const array& arr, int idx)
    : arr(arr), idx(idx) {
  if (arr.ndim() == 0) {
    throw std::invalid_argument("Cannot iterate over 0-d array.");
  }
}

array::ArrayIterator::reference array::ArrayIterator::operator*() const {
  auto start = Shape(arr.ndim(), 0);
  auto end = arr.shape();
  auto shape = arr.shape();
  shape.erase(shape.begin());
  start[0] = idx;
  end[0] = idx + 1;
  return reshape(slice(arr, start, end), shape);
};

} // namespace mlx::core
