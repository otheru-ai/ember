#pragma once

// Included from C++ (backend_dflash.cc) and from C (the regression test).
#ifndef __cplusplus
#include <stdbool.h>
#endif
// The wake condition for the continuous-batch coordinator loop in
// backend_dflash.cc, kept here as a pure function so it can be tested.
// backend_dflash.cc needs HIP and is not part of the host build, so a
// predicate written inline there is unreachable from the test suite; this
// header is the seam that makes the regression runnable.
//
// Two defects live in this one expression, both found the hard way.
//
// It must be a PREDICATE at all. batch_mu is released across pump(), so a
// submitter can enqueue and notify_one() in that window with nobody waiting.
// Condition variables do not latch, so the wakeup is dropped, and the loop
// tail does not re-read the queues. The unpredicated wait therefore slept
// forever on work already queued: that deadlocked gate 2 of the gfx1151
// certification twice, GPU at 0%, nothing ever submitted.
//
// It must also require ADMISSION CAPACITY for pending work. Waking merely
// because the pending queue is non-empty spins when the scheduler is at
// capacity -- terminal sessions retained for caller postprocessing hold
// their leases, the admission loop cannot admit, pump() returns Idle, and
// the wait returns immediately. Measured at ~2.3M pump calls per 100ms with
// two retained terminal sessions and one queued request. Waiting is correct
// there: a completion frees capacity and notifies.

static inline bool ember_batch_should_wake(bool has_controls,
                                           bool has_pending,
                                           bool stopping,
                                           int resident,
                                           int capacity) {
    if (stopping) return true;
    if (has_controls) return true;          // runs inline, needs no session slot
    return has_pending && resident < capacity;
}
