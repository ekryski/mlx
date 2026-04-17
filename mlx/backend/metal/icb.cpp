// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/icb.h"

#include <cstring>
#include <stdexcept>

#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

namespace {

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
  return dev->argumentBuffersSupport() >= MTL::ArgumentBuffersTier2;
}

IndirectCommandRecorder::IndirectCommandRecorder(
    Device& d,
    size_t max_commands_per_segment,
    size_t bytes_arena_cap)
    : device_(d),
      max_commands_per_segment_(max_commands_per_segment),
      bytes_arena_cap_(bytes_arena_cap) {
  if (max_commands_per_segment == 0) {
    throw std::invalid_argument(
        "[metal::IndirectCommandRecorder] max_commands_per_segment must be > 0");
  }

  auto* dev = device_.mtl_device();
  if (!dev) {
    throw std::runtime_error(
        "[metal::IndirectCommandRecorder] Metal device unavailable");
  }

  if (bytes_arena_cap_ > 0) {
    bytes_arena_ = NS::TransferPtr(
        dev->newBuffer(bytes_arena_cap_, MTL::ResourceStorageModeShared));
    if (!bytes_arena_) {
      throw std::runtime_error(
          "[metal::IndirectCommandRecorder] failed to allocate bytes arena");
    }
  }

  // Start with an empty first segment — commands_ is empty; the ICB will
  // be allocated lazily on split/finalize when we know the segment size.
  segments_.emplace_back();
  if (bytes_arena_) {
    segments_.back().resource_set.insert(bytes_arena_.get());
  }
  segments_.back().commands.reserve(max_commands_per_segment_);
}

IndirectCommandRecorder::~IndirectCommandRecorder() = default;

NS::SharedPtr<MTL::IndirectCommandBuffer>
IndirectCommandRecorder::allocate_icb_(size_t max_commands) {
  auto* dev = device_.mtl_device();
  auto desc =
      NS::TransferPtr(MTL::IndirectCommandBufferDescriptor::alloc()->init());
  desc->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
  desc->setInheritBuffers(false);
  desc->setInheritPipelineState(false);
  desc->setMaxKernelBufferBindCount(kMaxKernelBufferBindCount);
  auto icb = NS::TransferPtr(dev->newIndirectCommandBuffer(
      desc.get(), max_commands, MTL::ResourceStorageModePrivate));
  if (!icb) {
    throw std::runtime_error(
        "[metal::IndirectCommandRecorder] failed to allocate ICB segment");
  }
  return icb;
}

void IndirectCommandRecorder::materialize_icb_(
    const std::vector<Command>& commands,
    MTL::IndirectCommandBuffer* icb) {
  for (size_t i = 0; i < commands.size(); ++i) {
    const auto& c = commands[i];
    auto* icmd = icb->indirectComputeCommand(i);
    icmd->setComputePipelineState(c.pipeline);
    for (int slot = 0; slot < kMaxKernelBufferBindCount; ++slot) {
      const auto& b = c.bindings[slot];
      if (b.buffer) {
        icmd->setKernelBuffer(b.buffer, static_cast<NS::UInteger>(b.offset), slot);
      }
    }
    for (int j = 0; j < c.threadgroup_mem_count; ++j) {
      const auto& tm = c.threadgroup_mem[j];
      icmd->setThreadgroupMemoryLength(
          static_cast<NS::UInteger>(tm.length),
          static_cast<NS::UInteger>(tm.slot));
    }
    if (c.use_dispatch_threads) {
      icmd->concurrentDispatchThreads(c.grid, c.group);
    } else {
      icmd->concurrentDispatchThreadgroups(c.grid, c.group);
    }
  }
}

void IndirectCommandRecorder::track_resource_(const MTL::Buffer* buf) {
  if (!buf) {
    return;
  }
  // Recorder-wide retained set (to extend MTLBuffer lifetime). This
  // can be called any time — the retained set is global to the recorder.
  // Per-segment resource tracking happens in end_command, not here,
  // because a mid-command split could otherwise attribute buffers to the
  // wrong segment.
  if (all_retained_.insert(buf).second) {
    retained_buffers_.push_back(
        NS::RetainPtr(const_cast<MTL::Buffer*>(buf)));
  }
}

void IndirectCommandRecorder::begin_command(
    MTL::ComputePipelineState* pipeline) {
  if (finalized_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] cannot record after finalize");
  }
  if (segments_.back().commands.size() >= max_commands_per_segment_) {
    throw std::overflow_error(
        "[metal::IndirectCommandRecorder] segment capacity exceeded — "
        "split earlier or raise max_commands_per_segment");
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
  track_resource_(buf);
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
  // bytes_arena_ is already in the segment's resource_set (added at
  // construction or segment-start), so no track_resource_ needed here.
  return true;
}

void IndirectCommandRecorder::set_threadgroup_memory(size_t length, int slot) {
  if (!cur_active_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] set_threadgroup_memory outside command");
  }
  if (cur_.threadgroup_mem_count >= kMaxThreadgroupMemorySlots) {
    throw std::overflow_error(
        "[metal::IndirectCommandRecorder] threadgroup memory slots exceeded");
  }
  cur_.threadgroup_mem[cur_.threadgroup_mem_count++] =
      ThreadgroupMem{slot, length};
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

  // Attribute the command's buffers to the *current* segment's
  // resource_set. Done here (not in set_kernel_buffer) because a
  // split_segment() can fire between begin_command and end_command —
  // the command lands in whatever segment is current at end_command time.
  auto& seg = segments_.back();
  for (const auto& b : cur_.bindings) {
    if (b.buffer) {
      seg.resource_set.insert(b.buffer);
    }
  }

  seg.commands.push_back(cur_);
  ++total_commands_;
  cur_active_ = false;
  cur_ = Command{};
}

void IndirectCommandRecorder::split_segment() {
  if (finalized_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] split after finalize");
  }
  // Calling split while a command is partially built is intentional —
  // the split happens between the prior commands and the in-progress
  // one. `cur_` is held at the recorder level and stays intact across
  // segment boundaries; `end_command` will push it into whatever segment
  // is current at that moment.

  // Collapse no-op splits (empty segment) — the caller would otherwise
  // introduce a spurious memory barrier on an empty ICB.
  if (segments_.back().commands.empty()) {
    return;
  }

  // Allocate + materialize the just-closed segment's ICB now. Sized to
  // exactly the commands it holds.
  auto& seg = segments_.back();
  seg.icb = allocate_icb_(seg.commands.size());
  materialize_icb_(seg.commands, seg.icb.get());

  // Start a fresh segment. The bytes arena is shared across segments,
  // so include it in the new segment's resource_set.
  segments_.emplace_back();
  if (bytes_arena_) {
    segments_.back().resource_set.insert(bytes_arena_.get());
  }
  segments_.back().commands.reserve(max_commands_per_segment_);
}

void IndirectCommandRecorder::finalize() {
  if (finalized_) {
    return;
  }
  if (cur_active_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] pending command at finalize");
  }

  // Materialize any segments that don't have an ICB yet (the last one,
  // typically; possibly also segment 0 if only one segment exists).
  for (auto& seg : segments_) {
    if (!seg.icb && !seg.commands.empty()) {
      seg.icb = allocate_icb_(seg.commands.size());
      materialize_icb_(seg.commands, seg.icb.get());
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

  const auto usage = MTL::ResourceUsageRead | MTL::ResourceUsageWrite;
  bool any_emitted = false;
  for (const auto& seg : segments_) {
    if (seg.commands.empty() || !seg.icb) {
      continue;
    }
    if (any_emitted) {
      // Barrier between consecutive non-empty ICB segments.
      enc->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    for (const auto* buf : seg.resource_set) {
      enc->useResource(buf, usage);
    }
    enc->executeCommandsInBuffer(seg.icb.get(), NS::Range(0, seg.commands.size()));
    any_emitted = true;
  }
}

} // namespace mlx::core::metal
