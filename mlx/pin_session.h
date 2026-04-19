// Copyright © 2026 Apple Inc.

#pragma once

#include <cstddef>

#include "mlx/api.h"

// Stable-address allocator reuse — the foundation of decode-loop ICB
// Option (b). See array.cpp for the implementation and semantics.
//
// The PinSession struct is intentionally opaque at this layer; mlx-c
// and mlx-swift interact through the functions declared below, which
// take / return raw PinSession pointers.
namespace mlx::core::detail {

struct PinSession;

// Begin a pin session on the current thread in Record phase.
// Every `array::set_data` call on this thread captures its Data
// shared_ptr into the session. Returns the session handle; caller
// owns it and must release with `free_pin_session`.
// Throws if a session is already active on this thread.
MLX_API PinSession* begin_pin_record_session();

// End the current thread's record-phase session. Returns the number
// of slots captured. The session remains owned by the caller; the
// TLS pointer is cleared.
MLX_API size_t end_pin_record_session();

// Attach `sess` to the current thread in Replay phase. Subsequent
// `array::set_data` calls reuse the recorded slots in order.
// Throws if a session is already active on this thread.
MLX_API void begin_pin_replay(PinSession* sess);

// End the current thread's replay-phase session. Returns the number
// of slots consumed (should match the number of set_data calls on
// this pass — a mismatch vs slot count indicates graph divergence).
MLX_API size_t end_pin_replay();

// Release a session. Safe to call with nullptr. Do not call while
// the session is attached to any thread — defensively clears TLS
// if it matches, but the invariant is caller-managed.
MLX_API void free_pin_session(PinSession* sess);

// Diagnostic — total slots captured during Record.
MLX_API size_t pin_session_slot_count(PinSession* sess);

// Diagnostic — slots consumed so far in the current Replay.
MLX_API size_t pin_session_slot_idx(PinSession* sess);

} // namespace mlx::core::detail
