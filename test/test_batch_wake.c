// Regression for the continuous-batch coordinator wake condition.
// Both cases here were live defects, and the second was introduced by the fix
// for the first, so they are pinned together deliberately.
#include "backend/batch_wake.h"

#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static void check(const char *what, int got, int want) {
    if (got != want) {
        printf("FAIL %s: got %d want %d\n", what, got, want);
        failures++;
    }
}

int main(void) {
    // The deadlock: work queued while the coordinator was not waiting. The
    // dropped notify_one() is unrecoverable unless the predicate re-reads the
    // queues, so each queued kind must wake on its own.
    check("control queued wakes",
          ember_batch_should_wake(1, 0, 0, 0, 2), 1);
    check("pending queued wakes when capacity is free",
          ember_batch_should_wake(0, 1, 0, 0, 2), 1);
    check("stop wakes",
          ember_batch_should_wake(0, 0, 1, 0, 2), 1);

    // Nothing to do: waiting is correct. A wake here is the spin.
    check("idle sleeps",
          ember_batch_should_wake(0, 0, 0, 0, 2), 0);

    // The spin: pending work at full resident capacity. Terminal sessions
    // retained for caller postprocessing hold their leases, so the admission
    // loop cannot admit and pump() returns Idle. Waking would busy-loop;
    // a completion frees capacity and notifies.
    check("pending at full capacity sleeps",
          ember_batch_should_wake(0, 1, 0, 2, 2), 0);
    check("pending over capacity sleeps",
          ember_batch_should_wake(0, 1, 0, 3, 2), 0);

    // Controls run inline on the coordinator thread and take no session slot,
    // so capacity must not gate them -- gating them would reintroduce the
    // original deadlock for validation and disk control calls whenever the
    // scheduler happened to be full.
    check("control wakes even at full capacity",
          ember_batch_should_wake(1, 0, 0, 2, 2), 1);
    check("control wakes at full capacity with pending too",
          ember_batch_should_wake(1, 1, 0, 2, 2), 1);

    // Stop must win regardless, or shutdown hangs behind a full scheduler.
    check("stop wins at full capacity",
          ember_batch_should_wake(0, 1, 1, 2, 2), 1);

    // Degenerate capacity: a coordinator that can hold nothing must not spin
    // on queued work it can never admit.
    check("zero capacity sleeps on pending",
          ember_batch_should_wake(0, 1, 0, 0, 0), 0);

    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("batch wake predicate: all checks passed\n");
    return 0;
}
