// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/**
 * Reset the cumulative dispatch counter on the Metal command encoder of the
 * default device. Used to audit per-region dispatch counts — e.g. bracket a
 * decode-token forward pass with reset + total_dispatches to measure the
 * dispatch count that an ICB prototype would need to pre-encode.
 */
MLX_API void reset_dispatch_counter();

/**
 * Read the cumulative dispatch counter since the last reset. A dispatch is
 * either `dispatchThreadgroups` or `dispatchThreads` — i.e. one GPU kernel
 * launch.
 */
MLX_API uint64_t total_dispatches();

/**
 * Start recording the kernel name (pipeline state label) of every Metal
 * dispatch. Call once before a region of interest; every
 * dispatch-threadgroups / dispatch-threads between start and stop is
 * appended, in order, to an in-memory log.
 *
 * Enables the dispatch-list stability audit (compare kernel-name
 * sequences across tokens) that gates the Metal Indirect Command Buffer
 * work: identical sequences mean the dispatch list is stable and the
 * encode-once / execute-many ICB path is viable.
 *
 * Calling `start_kernel_log()` clears any prior log.
 */
MLX_API void start_kernel_log();

/** Stop recording kernel names. Idempotent. */
MLX_API void stop_kernel_log();

/** Number of entries in the current kernel log. */
MLX_API size_t kernel_log_size();

/**
 * Get the kernel label at index `i`, or empty string if out of range.
 * Returns a stable-over-the-log-lifetime C string; copy immediately if
 * you're going to call `start_kernel_log()` again.
 */
MLX_API const char* kernel_log_at(size_t i);

} // namespace mlx::core::metal
