// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/icb.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

namespace {

constexpr size_t kBytesArenaAlignment = 16;

size_t align_up(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

} // namespace

const std::vector<IndirectCommandRecorder::TagLocation>
    IndirectCommandRecorder::kEmptyTagLocations_{};

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

  if (const char* dbg = std::getenv("MLX_ICB_ORDER_TRACE"); dbg && dbg[0] == '1') {
    auto name = kernel_name_for(pipeline);
    if (name.empty()) {
      auto* lbl = pipeline->label();
      name = lbl ? std::string(lbl->utf8String()) : std::string("<unnamed>");
    }
    std::cerr << "[ICB-ORD] seg=" << (segments_.size() - 1)
              << " cmd=" << segments_.back().commands.size()
              << " " << name << "\n";
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

  // Resolve any pending binding tags against this command's bindings.
  // Done before push_back so we can index the command via segments_.size()-1
  // and seg.commands.size() as the final (post-push) position.
  const size_t seg_idx = segments_.size() - 1;
  const size_t cmd_idx = seg.commands.size();
  if (!pending_tags_by_buffer_.empty()) {
    for (int slot = 0; slot < kMaxKernelBufferBindCount; ++slot) {
      const auto& b = cur_.bindings[slot];
      if (!b.buffer) {
        continue;
      }
      auto it = pending_tags_by_buffer_.find(b.buffer);
      if (it == pending_tags_by_buffer_.end()) {
        continue;
      }
      for (uint32_t name_id : it->second) {
        tags_[name_id].push_back({seg_idx, cmd_idx, slot, b.offset});
      }
    }
  }

  // Buffer trace for ICB diagnostics — logs the first few commands'
  // bound buffers so we can see what the recorder captured. Gate on
  // MLX_ICB_BUFFER_TRACE=1.
  static const bool buf_trace_ = []() {
    const char* v = std::getenv("MLX_ICB_BUFFER_TRACE");
    return v && v[0] == '1';
  }();
  if (buf_trace_ && total_commands_ < 10) {
    std::cerr << "[ICB-BUF] cmd=" << total_commands_ << " bindings:";
    for (int slot = 0; slot < kMaxKernelBufferBindCount; ++slot) {
      const auto& b = cur_.bindings[slot];
      if (b.buffer) {
        std::cerr << " [" << slot << "=" << b.buffer
                  << "+" << b.offset << "]";
      }
    }
    std::cerr << "\n";
  }
  if (buf_trace_ && !pending_tags_by_buffer_.empty() && total_commands_ < 20) {
    std::cerr << "[ICB-BUF] pending_tags:";
    for (const auto& [buf, names] : pending_tags_by_buffer_) {
      std::cerr << " buf=" << buf << "(n_names=" << names.size() << ")";
    }
    std::cerr << "\n";
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

size_t IndirectCommandRecorder::tag_binding(
    uint32_t name_id,
    const MTL::Buffer* buf) {
  if (finalized_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] cannot tag after finalize");
  }
  if (!buf) {
    throw std::invalid_argument(
        "[metal::IndirectCommandRecorder] null buffer in tag_binding");
  }

  // Step 1: scan already-recorded bindings in all segments for matches.
  size_t matches = 0;
  auto& locs = tags_[name_id];
  for (size_t si = 0; si < segments_.size(); ++si) {
    const auto& seg = segments_[si];
    for (size_t ci = 0; ci < seg.commands.size(); ++ci) {
      const auto& cmd = seg.commands[ci];
      for (int slot = 0; slot < kMaxKernelBufferBindCount; ++slot) {
        const auto& b = cmd.bindings[slot];
        if (b.buffer == buf) {
          locs.push_back({si, ci, slot, b.offset});
          ++matches;
        }
      }
    }
  }

  // Step 2: register pending so future bindings of `buf` under this
  // recorder also accrue tags via end_command.
  pending_tags_by_buffer_[buf].push_back(name_id);

  return matches;
}

const std::vector<IndirectCommandRecorder::TagLocation>&
IndirectCommandRecorder::tags_for(uint32_t name_id) const {
  auto it = tags_.find(name_id);
  if (it == tags_.end()) {
    return kEmptyTagLocations_;
  }
  return it->second;
}

void IndirectCommandRecorder::replay_with_overrides(
    MTL::ComputeCommandEncoder* enc,
    const std::vector<
        std::tuple<uint32_t, const MTL::Buffer*, int64_t>>& overrides) {
  if (!finalized_) {
    throw std::logic_error(
        "[metal::IndirectCommandRecorder] replay_with_overrides before finalize");
  }
  if (!enc) {
    throw std::invalid_argument(
        "[metal::IndirectCommandRecorder] null encoder");
  }

  // Per-segment additional-resource set for useResource. We must cover
  // every override buffer on the live encoder even if it wasn't part of
  // the original seg.resource_set.
  std::vector<std::unordered_set<const MTL::Buffer*>> extra_resources(
      segments_.size());

  // Apply overrides: for each (name, buffer, offset) triple, walk every
  // TagLocation registered under that name and rewrite the ICB's
  // indirect compute command's slot binding. Missing names are skipped
  // silently — caller may choose to override only a subset.
  static const bool override_trace_ = []() {
    const char* v = std::getenv("MLX_ICB_OVERRIDE_TRACE");
    return v && v[0] == '1';
  }();
  for (const auto& [name_id, new_buf, new_offset] : overrides) {
    if (!new_buf) {
      throw std::invalid_argument(
          "[metal::IndirectCommandRecorder] null buffer in overrides");
    }
    auto it = tags_.find(name_id);
    if (override_trace_) {
      size_t n = (it == tags_.end()) ? 0 : it->second.size();
      std::cerr << "[ICB-OVERRIDE] name_id=" << name_id
                << " new_buf=" << new_buf
                << " new_offset=" << new_offset
                << " tag_locations=" << n << "\n";
    }
    if (it == tags_.end()) {
      continue;
    }
    for (const auto& loc : it->second) {
      auto& seg = segments_[loc.segment_idx];
      if (!seg.icb) {
        // Empty segment somehow registered a tag — skip defensively.
        continue;
      }
      auto* icmd = seg.icb->indirectComputeCommand(loc.command_in_segment);
      icmd->setKernelBuffer(
          new_buf,
          static_cast<NS::UInteger>(new_offset),
          loc.slot);
      extra_resources[loc.segment_idx].insert(new_buf);
    }
  }

  // Replay — identical structure to replay() but unions in the
  // per-segment override buffers when calling useResource.
  const auto usage = MTL::ResourceUsageRead | MTL::ResourceUsageWrite;
  bool any_emitted = false;
  for (size_t si = 0; si < segments_.size(); ++si) {
    const auto& seg = segments_[si];
    if (seg.commands.empty() || !seg.icb) {
      continue;
    }
    if (any_emitted) {
      enc->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    for (const auto* buf : seg.resource_set) {
      enc->useResource(buf, usage);
    }
    for (const auto* buf : extra_resources[si]) {
      enc->useResource(buf, usage);
    }
    enc->executeCommandsInBuffer(seg.icb.get(), NS::Range(0, seg.commands.size()));
    any_emitted = true;
  }
}

} // namespace mlx::core::metal
