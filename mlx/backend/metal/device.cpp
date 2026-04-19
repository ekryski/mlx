// Copyright © 2023-2024 Apple Inc.

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <sstream>

#include <fmt/format.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/icb.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace std {

// Required for putting the pointer in unordered_set.
template <class T>
struct hash<NS::SharedPtr<T>> {
  size_t operator()(const NS::SharedPtr<T>& p) const {
    return std::hash<T*>{}(p.get());
  }
};

} // namespace std

namespace mlx::core::metal {

namespace {

constexpr const char* default_mtllib_path = METAL_PATH;

auto get_metal_version() {
  auto get_metal_version_ = []() {
    if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)) {
      return MTL::LanguageVersion4_0;
    } else if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}

NS::SharedPtr<MTL::Device> load_device() {
  auto pool = new_scoped_memory_pool();
  auto devices = NS::TransferPtr(MTL::CopyAllDevices());
  auto device = NS::RetainPtr(static_cast<MTL::Device*>(devices->object(0)))
      ?: NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    throw std::runtime_error("Failed to load device");
  }
  return device;
}

std::pair<MTL::Library*, NS::Error*> load_library_from_path(
    MTL::Device* device,
    const char* path) {
  auto library = NS::String::string(path, NS::UTF8StringEncoding);
  NS::Error* error;
  auto lib = device->newLibrary(library, &error);

  return std::make_pair(lib, error);
}

#ifdef SWIFTPM_BUNDLE
MTL::Library* try_load_bundle(
    MTL::Device* device,
    NS::URL* url,
    const std::string& lib_name) {
  std::string bundle_path = std::string(url->fileSystemRepresentation()) + "/" +
      SWIFTPM_BUNDLE + ".bundle";
  auto bundle = NS::Bundle::alloc()->init(
      NS::String::string(bundle_path.c_str(), NS::UTF8StringEncoding));
  if (bundle != nullptr) {
    std::string resource_path =
        std::string(bundle->resourceURL()->fileSystemRepresentation()) + "/" +
        lib_name + ".metallib";
    auto [lib, error] = load_library_from_path(device, resource_path.c_str());
    if (lib) {
      return lib;
    }
  }
  return nullptr;
}

MTL::Library* try_load_framework(
    MTL::Device* device,
    NS::URL* url,
    const std::string& lib_name) {
  std::string resource_path = std::string(url->fileSystemRepresentation()) +
      "/" + lib_name + ".metallib";
  auto [lib, error] = load_library_from_path(device, resource_path.c_str());
  if (lib) {
    return lib;
  }
  return nullptr;
}
#endif

// Firstly, search for the metallib in the same path as this binary
std::pair<MTL::Library*, NS::Error*> load_colocated_library(
    MTL::Device* device,
    const std::string& relative_path) {
  auto path = current_binary_dir() / relative_path;
  if (!path.has_extension()) {
    path.replace_extension(".metallib");
  }

  return load_library_from_path(device, path.c_str());
}

std::pair<MTL::Library*, NS::Error*> load_swiftpm_library(
    MTL::Device* device,
    const std::string& lib_name) {
#ifdef SWIFTPM_BUNDLE
  MTL::Library* library =
      try_load_bundle(device, NS::Bundle::mainBundle()->bundleURL(), lib_name);
  if (library != nullptr) {
    return {library, nullptr};
  }
  auto bundles = NS::Bundle::allBundles();
  for (int i = 0, c = (int)bundles->count(); i < c; i++) {
    auto bundle = reinterpret_cast<NS::Bundle*>(bundles->object(i));
    library = try_load_bundle(device, bundle->resourceURL(), lib_name);
    if (library != nullptr) {
      return {library, nullptr};
    }
  }
  // if SWIFTPM_BUNDLE is a framework identifier, try loading from that
  auto frameworks = NS::Bundle::allFrameworks();
  for (int i = 0, c = (int)frameworks->count(); i < c; i++) {
    const auto bundle = reinterpret_cast<NS::Bundle*>(frameworks->object(i));
    const auto identifier = bundle->bundleIdentifier();
    if (identifier != nullptr &&
        !strcmp(identifier->utf8String(), SWIFTPM_BUNDLE)) {
      library = try_load_framework(device, bundle->resourceURL(), lib_name);
      if (library != nullptr) {
        return {library, nullptr};
      }
    }
  }
#endif
  return {nullptr, nullptr};
}

MTL::Library* load_default_library(MTL::Device* device) {
  NS::Error* error[5];
  MTL::Library* lib;
  // First try the colocated mlx.metallib
  std::tie(lib, error[0]) = load_colocated_library(device, "mlx");
  if (lib) {
    return lib;
  }

  std::tie(lib, error[1]) = load_colocated_library(device, "Resources/mlx");
  if (lib) {
    return lib;
  }

  // Then try default.metallib in a SwiftPM bundle if we have one
  std::tie(lib, error[2]) = load_swiftpm_library(device, "default");
  if (lib) {
    return lib;
  }

  // Try lo load resources from Framework resources if SwiftPM wrapped as a
  // dynamic framework.
  std::tie(lib, error[3]) = load_colocated_library(device, "Resources/default");
  if (lib) {
    return lib;
  }

  // Finally try default_mtllib_path
  std::tie(lib, error[4]) = load_library_from_path(device, default_mtllib_path);
  if (!lib) {
    std::ostringstream msg;
    msg << "Failed to load the default metallib. ";
    for (int i = 0; i < 5; i++) {
      if (error[i] != nullptr) {
        msg << error[i]->localizedDescription()->utf8String() << " ";
      }
    }
    throw std::runtime_error(msg.str());
  }
  return lib;
}

MTL::Library* load_library(
    MTL::Device* device,
    const std::string& lib_name,
    const std::string& lib_path) {
  // We have been given a path that ends in metallib so try to load it
  if (lib_path.size() > 9 &&
      std::equal(lib_path.end() - 9, lib_path.end(), ".metallib")) {
    auto [lib, error] = load_library_from_path(device, lib_path.c_str());
    if (!lib) {
      std::ostringstream msg;
      msg << "Failed to load the metallib from <" << lib_path << "> with error "
          << error->localizedDescription()->utf8String();
      throw std::runtime_error(msg.str());
    }
    return lib;
  }

  // We have been given a path so try to load from lib_path / lib_name.metallib
  if (lib_path.size() > 0) {
    std::string full_path = lib_path + "/" + lib_name + ".metallib";
    auto [lib, error] = load_library_from_path(device, full_path.c_str());
    if (!lib) {
      std::ostringstream msg;
      msg << "Failed to load the metallib from <" << full_path
          << "> with error " << error->localizedDescription()->utf8String();
      throw std::runtime_error(msg.str());
    }
    return lib;
  }

  // Try to load the colocated library
  {
    auto [lib, error] = load_colocated_library(device, lib_name);
    if (lib) {
      return lib;
    }
  }

  // Try to load the library from swiftpm
  {
    auto [lib, error] = load_swiftpm_library(device, lib_name);
    if (lib) {
      return lib;
    }
  }

  std::ostringstream msg;
  msg << "Failed to load the metallib " << lib_name << ".metallib. "
      << "We attempted to load it from <" << current_binary_dir() << "/"
      << lib_name << ".metallib>";
#ifdef SWIFTPM_BUNDLE
  msg << " and from the Swift PM bundle.";
#endif
  throw std::runtime_error(msg.str());
}

} // namespace

CommandEncoder::CommandEncoder(
    Device& d,
    int index,
    ResidencySet& residency_set)
    : device_(d) {
  auto pool = new_scoped_memory_pool();
  queue_ = NS::TransferPtr(device_.mtl_device()->newCommandQueue());
  if (!queue_) {
    throw std::runtime_error(
        "[metal::CommandEncoder] Failed to make new command queue.");
  }
  if (residency_set.mtl_residency_set()) {
    queue_->addResidencySet(residency_set.mtl_residency_set());
  }
  debug_set_stream_queue_label(queue_.get(), index);
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
}

CommandEncoder::~CommandEncoder() {
  exiting_ = true;
  synchronize();
  auto pool = new_scoped_memory_pool();
  buffer_.reset();
  queue_.reset();
}

void CommandEncoder::set_buffer(
    const MTL::Buffer* buf,
    int idx,
    int64_t offset /* = 0 */) {
  if (recording_) {
    if (!has_pending_command_) {
      // Matches the set_input_array policy: tolerate pre-pipeline staging.
      return;
    }
    // Raw MTLBuffer bindings bypass the dependency-tracking path (there's
    // no mlx::array to check for RAW against prev outputs). Just route to
    // the recorder.
    active_recorder_->set_kernel_buffer(buf, offset, idx);
    return;
  }
  // Record as both input and output to ensure synchronization between command
  // buffers
  all_inputs_.insert((void*)buf);
  all_outputs_.insert((void*)buf);
  get_command_encoder()->setBuffer(buf, offset, idx);
}

void CommandEncoder::set_input_array(
    const array& a,
    int idx,
    int64_t offset /* = 0 */) {
  // Barrier tracking runs whether recording or not — during recording
  // the flag drives ICB segment splitting via maybeInsertBarrier; outside
  // recording it drives memory-barrier emission on the live encoder.
  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  needs_barrier_ =
      needs_barrier_ | (prev_outputs_.find(r_buf) != prev_outputs_.end());
  if (recording_) {
    // Some primitives stage input arrays outside of a strict
    // pipeline→bind→dispatch sequence (e.g. pre-registering dependencies
    // for the barrier tracker). These calls still matter for barrier
    // splitting but are meaningless to the ICB (they'd bind to no command).
    // Swallow them silently when no command is in progress.
    if (!has_pending_command_) {
      icb_skipped_set_input_++;
      return;
    }
    auto* a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
    active_recorder_->set_kernel_buffer(a_buf, a.offset() + offset, idx);
    return;
  }
  if (all_inputs_.insert(a.buffer().ptr()).second) {
    buffer_sizes_ += a.data_size();
  }
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  get_command_encoder()->setBuffer(a_buf, a.offset() + offset, idx);
}

void CommandEncoder::set_output_array(
    array& a,
    int idx,
    int64_t offset /* = 0 */) {
  if (recording_) {
    // Track outputs in next_outputs_ so a subsequent read of this array
    // raises needs_barrier_ and triggers an ICB segment split.
    set_input_array(a, idx, offset);
    register_output_array(a);
    return;
  }
  // Add barriers before adding the output to the output set
  set_input_array(a, idx, offset);
  register_output_array(a);
}

void CommandEncoder::set_threadgroup_memory_length(size_t length, int idx) {
  if (recording_) {
    active_recorder_->set_threadgroup_memory(length, idx);
    return;
  }
  get_command_encoder()->setThreadgroupMemoryLength(length, idx);
}

void CommandEncoder::set_bytes_raw(const void* data, size_t length, int idx) {
  if (recording_) {
    if (!active_recorder_->set_bytes(data, length, idx)) {
      // Arena exhausted. Abort the whole recording session so the caller
      // sees a clean "not recording" state on rethrow and can retry with
      // a larger arena. Without this reset, `recording_` stays true and
      // every subsequent begin_icb_recording throws "already recording".
      abort_icb_recording_();
      throw std::runtime_error(
          "[metal::CommandEncoder] ICB bytes arena exhausted; "
          "pick a larger bytes_arena_cap on begin_icb_recording.");
    }
    return;
  }
  get_command_encoder()->setBytes(data, length, idx);
}

void CommandEncoder::use_resource(
    const MTL::Resource* res,
    MTL::ResourceUsage usage) {
  if (recording_) {
    // ICB recording path not supported for now; callers using AB +
    // ICB must arrange residency through heap declarations.
    return;
  }
  get_command_encoder()->useResource(res, usage);
}

void CommandEncoder::register_input_array(const array& a) {
  // Mirrors the dependency-tracking half of `set_input_array` but skips
  // the `setBuffer` emit. Used by primitives that bind their inputs
  // indirectly (e.g. via an ArgumentBuffer at buffer(0)) — the buffer
  // is still read by the GPU, so barrier + cross-encoder fencing must
  // still see it.
  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  needs_barrier_ =
      needs_barrier_ | (prev_outputs_.find(r_buf) != prev_outputs_.end());
  if (recording_) {
    return;
  }
  if (all_inputs_.insert(a.buffer().ptr()).second) {
    buffer_sizes_ += a.data_size();
  }
}

void CommandEncoder::register_output_array(const array& a) {
  all_outputs_.insert(a.buffer().ptr());

  auto buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  if (concurrent_) {
    concurrent_outputs_.insert(buf);
  } else {
    next_outputs_.insert(buf);
  }
}

void CommandEncoder::add_temporary(array arr) {
  temporaries_.push_back(std::move(arr));
}

void CommandEncoder::add_temporaries(std::vector<array> arrays) {
  temporaries_.insert(
      temporaries_.end(),
      std::make_move_iterator(arrays.begin()),
      std::make_move_iterator(arrays.end()));
}

void CommandEncoder::add_temporary_object(std::shared_ptr<void> obj) {
  temporary_objects_.push_back(std::move(obj));
}

void CommandEncoder::maybeInsertBarrier() {
  if (recording_) {
    // MTLIndirectComputeCommand cannot emit memory barriers. When a RAW
    // dependency is detected, split the recording into a new ICB segment
    // — replay will insert a memoryBarrier between segments on the live
    // encoder. Dependency tracking (prev_outputs_ / next_outputs_ /
    // needs_barrier_) runs the same as in the live path so splits fire
    // exactly where a live barrier would.
    if (needs_barrier_) {
      active_recorder_->split_segment();
      needs_barrier_ = false;
      prev_outputs_ = std::move(next_outputs_);
    } else {
      prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());
    }
    next_outputs_.clear();
    return;
  }
  if (needs_barrier_) {
    get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
    needs_barrier_ = false;
    prev_outputs_ = std::move(next_outputs_);
  } else {
    prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());
  }
  next_outputs_.clear();
}

// Process-wide dispatch counter. Must be static (not per-encoder) so the
// `metal::reset_dispatch_counter()` / `total_dispatches()` API returns the
// total across all stream encoders — MLX ops can run on any of several
// stream-specific encoders, and a per-encoder counter would miss anything
// not on whichever stream the caller happened to query.
static std::atomic<uint64_t> g_total_dispatches_{0};
// Opt-in flag. The atomic fetch_add on every dispatch is a fixed
// per-kernel tax (measurable on high-dispatch-count models like
// GPT-OSS 20B at ~1500 kernels/token); by default we skip the counter
// entirely. `reset_dispatch_counter()` enables it — a caller that
// wants the counter must reset it first.
static std::atomic<bool> g_dispatch_counter_enabled_{false};

// Kernel-name log: per-encoder thread-local "last kernel set" captured in
// set_compute_pipeline_state(), and a global mutex-protected vector
// appended to on each dispatch when the log is enabled. The enable flag
// is atomic so start/stop is cheap when off (one load).
static std::atomic<bool> g_kernel_log_enabled_{false};
static std::mutex g_kernel_log_mutex_;
static std::vector<std::string> g_kernel_log_;
// Last kernel label set on any encoder. Fine for single-threaded
// generation; for concurrent multi-stream workloads we'd want this
// per-encoder, but the ICB audit uses single-threaded decode.
static thread_local std::string g_last_kernel_label_;

// Map MTL::ComputePipelineState* → registration name, populated by
// Device::get_kernel_ at pipeline-build time. Already inside
// namespace mlx::core::metal, so these become mlx::core::metal::register_kernel_name
// etc — matching the declarations in the header.
static std::mutex g_kernel_name_mutex_;
static std::unordered_map<const MTL::ComputePipelineState*, std::string>
    g_kernel_name_map_;

void register_kernel_name(
    const MTL::ComputePipelineState* pipeline, const std::string& name) {
  std::lock_guard<std::mutex> lk(g_kernel_name_mutex_);
  g_kernel_name_map_[pipeline] = name;
}

std::string kernel_name_for(const MTL::ComputePipelineState* pipeline) {
  std::lock_guard<std::mutex> lk(g_kernel_name_mutex_);
  auto it = g_kernel_name_map_.find(pipeline);
  return (it == g_kernel_name_map_.end()) ? std::string("") : it->second;
}

void CommandEncoder::set_compute_pipeline_state(
    MTL::ComputePipelineState* kernel) {
  if (g_kernel_log_enabled_.load(std::memory_order_relaxed)) {
    auto name = kernel_name_for(kernel);
    if (name.empty()) {
      auto* label = kernel->label();
      name = label
          ? std::string(label->utf8String())
          : std::string("<unnamed>");
    }
    g_last_kernel_label_ = std::move(name);
  }
  if (recording_) {
    active_recorder_->begin_command(kernel);
    has_pending_command_ = true;
    return;
  }
  get_command_encoder()->setComputePipelineState(kernel);
}

void CommandEncoder::dispatch_threadgroups(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  if (recording_) {
    if (!has_pending_command_) {
      throw std::logic_error(
          "[metal::CommandEncoder] dispatch_threadgroups while recording "
          "without a preceding set_compute_pipeline_state");
    }
    // Barrier check fires *before* the command is appended to the current
    // segment, so when needs_barrier_ is true the current command lands in
    // a freshly-started segment — replay will emit a memoryBarrier on the
    // live encoder between the two segments.
    maybeInsertBarrier();
    active_recorder_->end_command(grid_dims, group_dims, /*use_threads=*/false);
    has_pending_command_ = false;
    icb_dispatch_calls_++;
    return;
  }
  maybeInsertBarrier();
  buffer_ops_++;
  if (g_dispatch_counter_enabled_.load(std::memory_order_relaxed)) {
    g_total_dispatches_.fetch_add(1, std::memory_order_relaxed);
  }
  if (g_kernel_log_enabled_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lk(g_kernel_log_mutex_);
    g_kernel_log_.push_back(g_last_kernel_label_);
  }
  get_command_encoder()->dispatchThreadgroups(grid_dims, group_dims);
}

void CommandEncoder::dispatch_threads(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  if (recording_) {
    if (!has_pending_command_) {
      throw std::logic_error(
          "[metal::CommandEncoder] dispatch_threads while recording "
          "without a preceding set_compute_pipeline_state");
    }
    maybeInsertBarrier();
    active_recorder_->end_command(grid_dims, group_dims, /*use_threads=*/true);
    has_pending_command_ = false;
    icb_dispatch_calls_++;
    return;
  }
  maybeInsertBarrier();
  buffer_ops_++;
  if (g_dispatch_counter_enabled_.load(std::memory_order_relaxed)) {
    g_total_dispatches_.fetch_add(1, std::memory_order_relaxed);
  }
  if (g_kernel_log_enabled_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lk(g_kernel_log_mutex_);
    g_kernel_log_.push_back(g_last_kernel_label_);
  }
  get_command_encoder()->dispatchThreads(grid_dims, group_dims);
}

void CommandEncoder::reset_dispatch_counter() {
  g_total_dispatches_.store(0, std::memory_order_relaxed);
  // Enable counting from this point on. Callers that don't care about
  // the counter (i.e. production decode) pay no per-dispatch atomic.
  g_dispatch_counter_enabled_.store(true, std::memory_order_relaxed);
}

uint64_t CommandEncoder::total_dispatches() const {
  return g_total_dispatches_.load(std::memory_order_relaxed);
}

void CommandEncoder::start_kernel_log() {
  {
    std::lock_guard<std::mutex> lk(g_kernel_log_mutex_);
    g_kernel_log_.clear();
  }
  g_kernel_log_enabled_.store(true, std::memory_order_relaxed);
}

void CommandEncoder::stop_kernel_log() {
  g_kernel_log_enabled_.store(false, std::memory_order_relaxed);
}

size_t CommandEncoder::kernel_log_size() {
  std::lock_guard<std::mutex> lk(g_kernel_log_mutex_);
  return g_kernel_log_.size();
}

const char* CommandEncoder::kernel_log_at(size_t i) {
  std::lock_guard<std::mutex> lk(g_kernel_log_mutex_);
  if (i >= g_kernel_log_.size()) return nullptr;
  return g_kernel_log_[i].c_str();
}

void CommandEncoder::barrier() {
  get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
}

void CommandEncoder::end_encoding() {
  // Each command encoder has a unique fence. We also store a map of
  // all previous outputs of command encoders to their corresponding fence.
  // - The command encoder records its inputs and outputs.
  // - Wait on a fence if any inputs in the encoder are outputs of a previous
  //   encoder.
  // - Update the map of outputs to include this command encoder's outputs.
  // - Always signal this command encoders fence.
  // - Add a completion handler for this command encoder that removes outputs
  //   from the map to limit the growth of the map and avoid unnecessary waits
  // - Temporaries are a special case as they do not cross command encoder
  //   boundaries. These can be removed early from the encoders inputs and
  //   outputs since they don't need synchronization.
  if (!encoder_) {
    return;
  }

  // Remove temporaries from inputs and outputs.
  for (auto& t : temporaries_) {
    all_outputs_.erase(t.buffer().ptr());
    all_inputs_.erase(t.buffer().ptr());
  }

  // Keep references to the fences we waited on and put them in the completion
  // handler so they are not prematurely released.
  std::unordered_set<NS::SharedPtr<MTL::Fence>> waiting_on;
  {
    std::lock_guard lk(outputs_mtx_);
    for (auto& in : all_inputs_) {
      if (auto it = prev_ce_outputs_.find(in); it != prev_ce_outputs_.end()) {
        // If we've already waited on a fence, don't wait on it again.
        if (waiting_on.find(it->second) == waiting_on.end()) {
          encoder_->waitForFence(it->second.get());
          waiting_on.insert(it->second);
        }
      }
    }
    for (auto& out : all_outputs_) {
      prev_ce_outputs_[out] = fence_;
    }
  }

  encoder_->updateFence(fence_.get());
  buffer_->addCompletedHandler([this,
                                fence = std::move(fence_),
                                temporaries = std::move(temporaries_),
                                temporary_objects = std::move(temporary_objects_),
                                all_outputs = std::move(all_outputs_),
                                waiting_on = std::move(waiting_on)](
                                   MTL::CommandBuffer*) mutable {
    std::lock_guard lk(outputs_mtx_);
    for (auto& o : all_outputs) {
      if (auto it = prev_ce_outputs_.find(o); it != prev_ce_outputs_.end()) {
        if (it->second == fence) {
          prev_ce_outputs_.erase(it);
        }
      }
    }
  });

  encoder_->endEncoding();
  encoder_.reset();
  needs_barrier_ = false;
  concurrent_ = false;
  prev_outputs_.clear();
  next_outputs_.clear();
  concurrent_outputs_.clear();
  all_inputs_.clear();
}

bool CommandEncoder::needs_commit() const {
  auto [max_ops, max_mb] = device_.get_max_ops_mb_per_buffer();
  return (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb);
}

void CommandEncoder::commit() {
  buffer_->commit();
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
  buffer_ops_ = 0;
  buffer_sizes_ = 0;
}

void CommandEncoder::synchronize() {
  auto pool = new_scoped_memory_pool();
  auto cb = NS::RetainPtr(get_command_buffer());
  end_encoding();
  commit();
  cb->waitUntilCompleted();
  if (!exiting_) {
    if (cb->status() == MTL::CommandBufferStatusError) {
      throw std::runtime_error(
          fmt::format(
              "[METAL] Command buffer execution failed: {}.",
              cb->error()->localizedDescription()->utf8String()));
    }
  }
}

// Thread-local pointer set by begin_icb_recording so that
// get_command_encoder(Stream) redirects every stream's lookup to the
// recording encoder for the duration of the capture. Declared extern
// because `get_command_encoder` is defined later in the file.
extern thread_local CommandEncoder* t_icb_steer_target_;

void CommandEncoder::begin_icb_recording(
    size_t max_commands,
    size_t bytes_arena_cap) {
  if (recording_) {
    throw std::logic_error(
        "[metal::CommandEncoder] begin_icb_recording while already recording");
  }
  if (t_icb_steer_target_ != nullptr) {
    throw std::logic_error(
        "[metal::CommandEncoder] begin_icb_recording while a different recording "
        "is steering this thread");
  }
  // Flush anything the live encoder had pending so it doesn't interleave
  // with the ICB we're about to build. The live encoder is left in a
  // clean state and will be lazily recreated on the next direct call.
  end_encoding();

  active_recorder_ = std::make_unique<IndirectCommandRecorder>(
      device_, max_commands, bytes_arena_cap);
  recording_ = true;
  has_pending_command_ = false;
  icb_skipped_set_input_ = 0;
  icb_dispatch_calls_ = 0;
  ab_tag_counter_ = 0;

  // Steer every stream's encoder lookup on this thread through us until
  // end / abort. A secondary-stream primitive (MoE expert gather, a
  // fast kernel taking a caller-supplied Stream) will now call through
  // our set_compute_pipeline_state / set_input_array / dispatch_* and
  // accumulate into the recorder instead of emitting live on a sibling
  // encoder.
  t_icb_steer_target_ = this;
}

std::unique_ptr<IndirectCommandRecorder>
CommandEncoder::end_icb_recording() {
  if (!recording_) {
    throw std::logic_error(
        "[metal::CommandEncoder] end_icb_recording without begin");
  }
  if (has_pending_command_) {
    // Bad caller state, but drop recording so they aren't trapped in it.
    abort_icb_recording_();
    throw std::logic_error(
        "[metal::CommandEncoder] end_icb_recording with a pending command "
        "(missing dispatch after set_compute_pipeline_state)");
  }
  // Expose recording diagnostics so callers can reconcile the ICB size
  // against the live GPU.totalDispatches() counter.
  std::cerr
      << "[metal::ICB] dispatch calls routed through encoder during recording: "
      << icb_dispatch_calls_
      << "; set_input/buffer pre-pipeline skips: " << icb_skipped_set_input_
      << std::endl;
  if (icb_skipped_set_input_ > 0) {
    std::cerr
        << "[metal::ICB] warning: "
        << icb_skipped_set_input_
        << " set_input_array / set_buffer calls were skipped during recording "
           "because no pipeline was bound. Replay may be missing bindings."
        << std::endl;
  }
  active_recorder_->finalize();
  recording_ = false;
  t_icb_steer_target_ = nullptr;
  auto r = std::move(active_recorder_);
  return r;
}

void CommandEncoder::abort_icb_recording_() {
  recording_ = false;
  active_recorder_.reset();
  has_pending_command_ = false;
  if (t_icb_steer_target_ == this) {
    t_icb_steer_target_ = nullptr;
  }
}

void CommandEncoder::abort_icb_recording() {
  abort_icb_recording_();
}

void CommandEncoder::replay_icb(const IndirectCommandRecorder& recorder) {
  if (recording_) {
    throw std::logic_error(
        "[metal::CommandEncoder] replay_icb while recording is active");
  }
  // Replay counts as a single "op" from the command-buffer-commit
  // threshold's perspective — even though the ICB may hold thousands of
  // dispatches, Metal schedules them as one work item on the live encoder.
  buffer_ops_ += 1;
  recorder.replay(get_command_encoder());
}

void CommandEncoder::tag_binding(uint32_t name_id, const array& a) {
  tag_binding(
      name_id, static_cast<const MTL::Buffer*>(a.buffer().ptr()));
}

void CommandEncoder::tag_binding(uint32_t name_id, const MTL::Buffer* buf) {
  if (!recording_) {
    throw std::logic_error(
        "[metal::CommandEncoder] tag_binding requires active recording");
  }
  active_recorder_->tag_binding(name_id, buf);
}

uint32_t CommandEncoder::tag_ab_binding(const MTL::Buffer* buf) {
  if (!recording_) {
    return 0;
  }
  // Sequential IDs starting at 1 so callers can use 0 as a sentinel
  // for "not tagged" if they ever care. The decode-loop ICB
  // orchestrator (mlx-swift-lm side) needs to call into this same
  // encoder during a "build-only" forward pass to obtain the matching
  // sequence at replay time.
  uint64_t next = ++ab_tag_counter_;
  uint32_t id = static_cast<uint32_t>(next);
  if (id == 0) {
    // Wrapped past 2^32; bump to 1 so 0 stays the sentinel. A single
    // recording session would have to bind 4B AB transients to hit
    // this — diagnostic, not a real concern.
    ab_tag_counter_ = 1;
    id = 1;
  }
  active_recorder_->tag_binding(id, buf);
  return id;
}

void CommandEncoder::replay_icb_with_overrides(
    IndirectCommandRecorder& recorder,
    const std::vector<
        std::tuple<uint32_t, const MTL::Buffer*, int64_t>>& overrides) {
  if (recording_) {
    throw std::logic_error(
        "[metal::CommandEncoder] replay_icb_with_overrides while recording is active");
  }

  // Input-side barrier check: for each override buffer recently
  // written by a live dispatch, flag `needs_barrier_` so a
  // memoryBarrier is emitted before the replay executes. Without
  // this, an ICB that reads an override buffer (e.g. the input
  // hidden state produced by the previous layer's live post)
  // races with the still-in-flight producer write.
  for (const auto& [name, buf, offset] : overrides) {
    (void)name;
    (void)offset;
    if (!buf) continue;
    auto* r =
        static_cast<MTL::Resource*>(const_cast<MTL::Buffer*>(buf));
    if (prev_outputs_.find(r) != prev_outputs_.end()) {
      needs_barrier_ = true;
    }
  }
  if (needs_barrier_) {
    get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
    needs_barrier_ = false;
    prev_outputs_ = std::move(next_outputs_);
  } else {
    prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());
  }
  next_outputs_.clear();

  buffer_ops_ += 1;
  recorder.replay_with_overrides(get_command_encoder(), overrides);

  // Output-side: mark every override buffer as a recent output so
  // a subsequent primitive reading one of them triggers a
  // memoryBarrier via the existing `needs_barrier_` /
  // `prev_outputs_` path. Marking all overrides (including
  // input-only ones) is conservative: an extra barrier on a
  // read-only buffer is harmless; a missed barrier on a write
  // buffer is a data race.
  for (const auto& [name, buf, offset] : overrides) {
    (void)name;
    (void)offset;
    if (!buf) continue;
    next_outputs_.insert(
        static_cast<MTL::Resource*>(const_cast<MTL::Buffer*>(buf)));
  }
}

MTL::ComputeCommandEncoder* CommandEncoder::get_command_encoder() {
  if (!encoder_) {
    encoder_ = NS::RetainPtr(
        buffer_->computeCommandEncoder(MTL::DispatchTypeConcurrent));
    fence_ = NS::TransferPtr(device_.mtl_device()->newFence());
  }
  return encoder_.get();
}

Device::Device() : device_(load_device()), residency_set_(device_.get()) {
  auto pool = new_scoped_memory_pool();
  default_library_ = NS::TransferPtr(load_default_library(device_.get()));
  arch_ = env::metal_gpu_arch();
  if (arch_.empty()) {
    arch_ = std::string(device_->architecture()->name()->utf8String());
  }
  int ag_tens = 0;
  int ag_ones = 0;
  if (arch_.size() >= 3) {
    ag_tens = arch_[arch_.size() - 3] - '0';
    ag_ones = arch_[arch_.size() - 2] - '0';
    ag_tens = (ag_tens < 10 && ag_tens >= 0) ? ag_tens : 0;
    ag_ones = (ag_ones < 10 && ag_ones >= 0) ? ag_ones : 0;
  }
  arch_gen_ = ag_tens * 10 + ag_ones;
  auto arch = arch_.back();
  switch (arch) {
    case 'p': // phone
      max_ops_per_buffer_ = 20;
      max_mb_per_buffer_ = 40;
      break;
    case 'g': // base, pro
      max_ops_per_buffer_ = 40;
      max_mb_per_buffer_ = 40;
      break;
    case 's': // max
      // Note: ops_per_buffer=300 gives +11% decode speed but increases peak
      // memory during prefill (more intermediates alive simultaneously).
      // Default 200 balances decode throughput vs peak prefill memory; raise
      // with MLX_MAX_OPS_PER_BUFFER for decode-heavy workloads when memory
      // allows.
      max_ops_per_buffer_ = 200;
      max_mb_per_buffer_ = 50;
      break;
    case 'd': // ultra
      max_ops_per_buffer_ = 200;
      max_mb_per_buffer_ = 50;
      break;
    default: // default to medium
      max_ops_per_buffer_ = 40;
      max_mb_per_buffer_ = 40;
      break;
  }
  max_ops_per_buffer_ = env::max_ops_per_buffer(max_ops_per_buffer_);
  max_mb_per_buffer_ = env::max_mb_per_buffer(max_mb_per_buffer_);
}

Device::~Device() = default;

MTL::Library* Device::get_library(
    const std::string& name,
    const std::string& path /* = "" */) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second.get();
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second.get();
  }

  auto new_lib = load_library(device_.get(), name, path.c_str());
  library_map_.insert({name, NS::TransferPtr(new_lib)});
  return new_lib;
}

NS::SharedPtr<MTL::Library> Device::build_library_(
    const std::string& source_string) {
  auto pool = new_scoped_memory_pool();

  auto ns_code =
      NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);

  NS::Error* error = nullptr;
  auto options = MTL::CompileOptions::alloc()->init()->autorelease();
  options->setFastMathEnabled(false);
  options->setLanguageVersion(get_metal_version());
#ifndef NDEBUG
  if (options->languageVersion() >= MTL::LanguageVersion3_2) {
    options->setEnableLogging(true);
  }
#endif
  auto mtl_lib = NS::TransferPtr(device_->newLibrary(ns_code, options, &error));

  // Throw error if unable to compile library
  if (!mtl_lib) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to build metal library from source\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_lib;
}

NS::SharedPtr<MTL::Function> Device::get_function_(
    const std::string& name,
    MTL::Library* mtl_lib) {
  auto pool = new_scoped_memory_pool();
  // Pull kernel from library
  auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  return NS::TransferPtr(mtl_lib->newFunction(ns_name));
}

NS::SharedPtr<MTL::Function> Device::get_function_(
    const std::string& name,
    const std::string& specialized_name,
    const MTLFCList& func_consts,
    MTL::Library* mtl_lib) {
  if (func_consts.empty() && (specialized_name == name)) {
    return get_function_(name, mtl_lib);
  }

  auto pool = new_scoped_memory_pool();

  // Prepare function constants
  auto mtl_func_consts =
      MTL::FunctionConstantValues::alloc()->init()->autorelease();

  for (auto [value, type, index] : func_consts) {
    mtl_func_consts->setConstantValue(value, type, index);
  }

  // Prepare function desc
  auto desc = MTL::FunctionDescriptor::functionDescriptor();
  desc->setName(NS::String::string(name.c_str(), NS::ASCIIStringEncoding));
  desc->setSpecializedName(
      NS::String::string(specialized_name.c_str(), NS::ASCIIStringEncoding));
  desc->setConstantValues(mtl_func_consts);

  // Pull kernel from library
  NS::Error* error = nullptr;
  auto mtl_function = NS::TransferPtr(mtl_lib->newFunction(desc, &error));

  // Throw error if unable to build metal function
  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load function " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_function;
}

// Opt-in for MTLComputePipelineDescriptor.supportIndirectCommandBuffers
// = true. Despite Apple's documentation claiming no cost for direct
// dispatch, empirical measurement on GPT-OSS 20B (~1500 kernels/token)
// shows a real per-dispatch tax on Apple Silicon. Default is off;
// callers that want ICB capture either set MLX_METAL_ICB=1 before
// first kernel compile, or call `set_icb_pipeline_support(true)`
// programmatically (tests). The programmatic override wins over env.
static std::atomic<int> g_icb_pipeline_support_override_{-1}; // -1 = use env

static bool icb_pipeline_flag_enabled_() {
  int override = g_icb_pipeline_support_override_.load(std::memory_order_relaxed);
  if (override >= 0) {
    return override != 0;
  }
  const char* v = std::getenv("MLX_METAL_ICB");
  return v != nullptr && v[0] == '1';
}

void set_icb_pipeline_support(bool enabled) {
  g_icb_pipeline_support_override_.store(
      enabled ? 1 : 0, std::memory_order_relaxed);
}

NS::SharedPtr<MTL::ComputePipelineState> Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function) {
  NS::Error* error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> kernel;

  if (mtl_function) {
    if (icb_pipeline_flag_enabled_()) {
      // ICB-capable path — slightly more expensive direct dispatch, but
      // a pipeline built without this flag cannot participate in an
      // IndirectCommandRecorder capture at all (Metal requires the flag
      // at pipeline creation time).
      auto pool = new_scoped_memory_pool();
      auto desc =
          MTL::ComputePipelineDescriptor::alloc()->init()->autorelease();
      desc->setComputeFunction(mtl_function);
      desc->setSupportIndirectCommandBuffers(true);
      kernel = NS::TransferPtr(device_->newComputePipelineState(
          desc, MTL::PipelineOptionNone, nullptr, &error));
    } else {
      // Fast path — direct function-to-PSO. Matches stock mlx behavior.
      kernel = NS::TransferPtr(
          device_->newComputePipelineState(mtl_function, &error));
    }
  }

  // Throw error if unable to compile metal function
  if (!mtl_function || !kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  // Register the pipeline → name so the dispatch-counter kernel log can
  // report meaningful names (MTLComputePipelineState has no writable
  // label and no accessor for the source function name).
  register_kernel_name(kernel.get(), name);

  return kernel;
}

NS::SharedPtr<MTL::ComputePipelineState> Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function,
    const MTL::LinkedFunctions* linked_functions) {
  // Check inputs
  if (!linked_functions) {
    return get_kernel_(name, mtl_function);
  }

  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    throw std::runtime_error(msg.str());
  }

  auto pool = new_scoped_memory_pool();

  // Prepare compute pipeline state descriptor
  auto desc = MTL::ComputePipelineDescriptor::alloc()->init()->autorelease();
  desc->setComputeFunction(mtl_function);
  desc->setLinkedFunctions(linked_functions);
  if (icb_pipeline_flag_enabled_()) {
    desc->setSupportIndirectCommandBuffers(true);
  }

  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  auto kernel = NS::TransferPtr(device_->newComputePipelineState(
      desc, MTL::PipelineOptionNone, nullptr, &error));

  // Throw error if unable to compile metal function
  if (!kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

MTL::Library* Device::get_library(
    const std::string& name,
    const std::function<std::string(void)>& builder) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second.get();
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second.get();
  }

  auto mtl_lib = build_library_(builder());
  library_map_.insert({name, mtl_lib});
  return mtl_lib.get();
}

void Device::clear_library(const std::string& name) {
  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    library_kernels_.erase(it->second.get());
    library_map_.erase(it);
  }
}

NS::SharedPtr<MTL::LinkedFunctions> Device::get_linked_functions_(
    const std::vector<MTL::Function*>& funcs) {
  if (funcs.empty()) {
    return nullptr;
  }

  auto pool = new_scoped_memory_pool();
  auto lfuncs = NS::TransferPtr(MTL::LinkedFunctions::linkedFunctions());
  NS::Array* funcs_arr = NS::Array::array(
      reinterpret_cast<const NS::Object* const*>(funcs.data()), funcs.size());
  lfuncs->setPrivateFunctions(funcs_arr);
  return lfuncs;
}

MTL::ComputePipelineState* Device::get_kernel_(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  // Single writer allowed
  std::unique_lock wlock(kernel_mtx_);

  // Try loading again to avoid loading twice
  auto& kernel_map_ = library_kernels_[mtl_lib];
  if (auto it = kernel_map_.find(hash_name); it != kernel_map_.end()) {
    return it->second.get();
  }

  auto pool = new_scoped_memory_pool();

  // Pull kernel from library
  auto mtl_function = get_function_(base_name, hash_name, func_consts, mtl_lib);

  // Compile kernel to compute pipeline
  auto mtl_linked_funcs = get_linked_functions_(linked_functions);
  auto kernel =
      get_kernel_(hash_name, mtl_function.get(), mtl_linked_funcs.get());

  // Add kernel to cache
  kernel_map_.insert({hash_name, kernel});

  return kernel.get();
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name /* = "" */,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  const auto& kname = hash_name.empty() ? base_name : hash_name;
  {
    // Multiple readers allowed
    std::shared_lock lock(kernel_mtx_);

    // Look for cached kernel
    auto& kernel_map_ = library_kernels_[mtl_lib];
    if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
      return it->second.get();
    }
  }
  return get_kernel_(base_name, mtl_lib, kname, func_consts, linked_functions);
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    const std::string& hash_name /*  = "" */,
    const MTLFCList& func_consts /*  = {} */,
    const std::vector<MTL::Function*>& linked_functions /*  = {} */) {
  return get_kernel(
      base_name,
      default_library_.get(),
      hash_name,
      func_consts,
      linked_functions);
}

Device& device(mlx::core::Device) {
  // Leak singleton device intentionally, to avoid cases where a compute kernel
  // returns and tries to access the object after it has been freed by the main
  // thread teardown.
  static Device* metal_device = new Device;
  return *metal_device;
}

// ─────────────────────────────────────────────────────────────────────
// ICB multi-stream capture: during an active recording, redirect every
// `get_command_encoder(Stream)` lookup to a single target encoder so
// that primitives which would normally dispatch to a secondary stream
// (e.g. MoE expert gather, fast primitives that take a caller-supplied
// stream) still accumulate into the recording encoder.
//
// The target pointer lives in thread_local storage — same scope as the
// encoder map itself — so concurrent recordings on different threads
// don't interfere, and the pointer never dangles across thread_local
// destruction.
//
// Lifecycle: set by `CommandEncoder::begin_icb_recording`, cleared by
// `end_icb_recording` / `abort_icb_recording_`.
// ─────────────────────────────────────────────────────────────────────
thread_local CommandEncoder* t_icb_steer_target_ = nullptr;

CommandEncoder& get_command_encoder(Stream s) {
  // If an ICB recording is active on this thread, every stream's
  // dispatches route to the recording encoder so primitives that
  // target secondary streams (MoE expert scheduling, fast kernels
  // taking caller-supplied streams) still land in the ICB instead of
  // bypassing it via a sibling encoder.
  if (t_icb_steer_target_ != nullptr) {
    return *t_icb_steer_target_;
  }
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    // Lazily initialize the command encoder for this stream on the current thread.
    // This handles Swift structured concurrency where Tasks can migrate between
    // threads, and the thread-local encoder map may not have the stream registered.
    auto& d = device(s.device);
    it = encoders.try_emplace(s.index, d, s.index, d.residency_set()).first;
  }
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool() {
  return NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
}

bool is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = metal::device(mlx::core::Device::gpu);
    auto arch = d.get_architecture().back();
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
#endif
}

} // namespace mlx::core::metal
