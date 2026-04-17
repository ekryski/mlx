// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/icb.h"

#include <cstring>
#include <stdexcept>

#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

namespace {

// Round up to 16-byte alignment — Metal requires argument alignment for
// most primitive types; 16 is safe for any simd-packed struct mlx passes.
constexpr size_t kBytesArenaAlignment = 16;

size_t align_up(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

} // namespace

bool IndirectCommandRecorder::is_supported(Device& d) {
  auto* dev = d.mtl_device();
  if (!dev) {
    return false;
  }
  // Argument buffer tier 2 is the requirement for ICB-backed compute on
  // Apple Silicon; every M-series chip meets this. Probe rather than
  // hard-coding to stay correct on future minimum-spec changes.
  return dev->argumentBuffersSupport() >= MTL::ArgumentBuffersTier2;
}

IndirectCommandRecorder::IndirectCommandRecorder(
    Device& d,
    size_t max_commands,
    size_t bytes_arena_cap)
    : device_(d), max_commands_(max_commands), bytes_arena_cap_(bytes_arena_cap) {
  if (max_commands == 0) {
    throw std::invalid_argument(
        "[metal::IndirectCommandRecorder] max_commands must be > 0");
  }

  auto* dev = device_.mtl_device();
  if (!dev) {
    throw std::runtime_error(
        "[metal::IndirectCommandRecorder] Metal device unavailable");
  }

  // Descriptor: per-command pipeline + up to kMaxKernelBufferBindCount
  // buffer bindings, concurrent-dispatch threadgroups. `inheritBuffers`
  // and `inheritPipelineState` are both false because every command we
  // record sets its own pipeline and bindings explicitly.
  auto desc =
      NS::TransferPtr(MTL::IndirectCommandBufferDescriptor::alloc()->init());
  desc->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
  desc->setInheritBuffers(false);
  desc->setInheritPipelineState(false);
  desc->setMaxKernelBufferBindCount(kMaxKernelBufferBindCount);

  icb_ = NS::TransferPtr(dev->newIndirectCommandBuffer(
      desc.get(), max_commands, MTL::ResourceStorageModePrivate));
  if (!icb_) {
    throw std::runtime_error(
        "[metal::IndirectCommandRecorder] failed to allocate ICB");
  }

  if (bytes_arena_cap_ > 0) {
    bytes_arena_ = NS::TransferPtr(
        dev->newBuffer(bytes_arena_cap_, MTL::ResourceStorageModeShared));
    if (!bytes_arena_) {
      throw std::runtime_error(
          "[metal::IndirectCommandRecorder] failed to allocate bytes arena");
    }
    // Arena always appears in the resource set so replay() useResources it.
    resource_set_.insert(bytes_arena_.get());
  }

  commands_.reserve(max_commands);
}

IndirectCommandRecorder::~IndirectCommandRecorder() = default;

void IndirectCommandRecorder::begin_command(
    MTL::ComputePipelineState* pipeline) {
  if (finalized_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] cannot record after finalize");
  }
  if (next_cmd_ >= max_commands_) {
    throw std::overflow_error(
        "[metal::IndirectCommandRecorder] capacity exceeded");
  }
  if (!pipeline) {
    throw std::invalid_argument(
        "[metal::IndirectCommandRecorder] null pipeline");
  }

  cur_ = Command{};
  cur_.pipeline = pipeline;
  cur_active_ = true;
}

void IndirectCommandRecorder::set_kernel_buffer(
    const MTL::Buffer* buf,
    int64_t offset,
    int slot) {
  if (!cur_active_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] set_kernel_buffer outside command");
  }
  if (slot < 0 || slot >= kMaxKernelBufferBindCount) {
    throw std::out_of_range(
        "[metal::IndirectCommandRecorder] buffer slot out of range");
  }
  cur_.bindings[slot] = Binding{buf, offset};
  if (buf) {
    resource_set_.insert(buf);
  }
}

bool IndirectCommandRecorder::set_bytes(
    const void* data,
    size_t length,
    int slot) {
  if (!cur_active_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] set_bytes outside command");
  }
  if (slot < 0 || slot >= kMaxKernelBufferBindCount) {
    throw std::out_of_range(
        "[metal::IndirectCommandRecorder] bytes slot out of range");
  }
  if (!bytes_arena_) {
    return false;
  }

  size_t aligned_off = align_up(bytes_offset_, kBytesArenaAlignment);
  if (aligned_off + length > bytes_arena_cap_) {
    return false;
  }

  auto* arena_base = static_cast<uint8_t*>(bytes_arena_->contents());
  std::memcpy(arena_base + aligned_off, data, length);

  cur_.bindings[slot] = Binding{bytes_arena_.get(), static_cast<int64_t>(aligned_off)};
  bytes_offset_ = aligned_off + length;
  return true;
}

void IndirectCommandRecorder::end_command(
    MTL::Size grid,
    MTL::Size group,
    bool use_dispatch_threads) {
  if (!cur_active_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] end_command without begin");
  }
  cur_.grid = grid;
  cur_.group = group;
  cur_.use_dispatch_threads = use_dispatch_threads;
  commands_.push_back(cur_);
  ++next_cmd_;
  cur_active_ = false;
  cur_ = Command{};
}

void IndirectCommandRecorder::finalize() {
  if (finalized_) {
    return;
  }
  if (cur_active_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] pending command at finalize");
  }

  for (size_t i = 0; i < commands_.size(); ++i) {
    const auto& c = commands_[i];
    auto* icmd = icb_->indirectComputeCommand(i);
    icmd->setComputePipelineState(c.pipeline);
    for (int slot = 0; slot < kMaxKernelBufferBindCount; ++slot) {
      const auto& b = c.bindings[slot];
      if (b.buffer) {
        icmd->setKernelBuffer(b.buffer, static_cast<NS::UInteger>(b.offset), slot);
      }
    }
    if (c.use_dispatch_threads) {
      icmd->concurrentDispatchThreads(c.grid, c.group);
    } else {
      icmd->concurrentDispatchThreadgroups(c.grid, c.group);
    }
  }

  finalized_ = true;
}

void IndirectCommandRecorder::replay(MTL::ComputeCommandEncoder* enc) const {
  if (!finalized_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] replay before finalize");
  }
  if (!enc) {
    throw std::invalid_argument(
        "[metal::IndirectCommandRecorder] null encoder");
  }
  if (commands_.empty()) {
    return;
  }

  // Residency: Metal requires every buffer the ICB touches to be declared
  // via useResource on the executing compute encoder. ResourceUsageRead |
  // Write is safe for any kernel signature — overspecifying is harmless.
  const auto usage = MTL::ResourceUsageRead | MTL::ResourceUsageWrite;
  for (const auto* buf : resource_set_) {
    enc->useResource(buf, usage);
  }

  enc->executeCommandsInBuffer(icb_.get(), NS::Range(0, commands_.size()));
}

} // namespace mlx::core::metal
