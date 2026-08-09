#include "deepseek4_dspark_scheduler.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using dflash::common::DSparkProfitScheduler;
using dflash::common::DSparkBatchVerifyGate;
using dflash::common::DSparkSchedulerConfig;
using dflash::common::dspark_request_can_prepare;
using dflash::common::kDSparkMinSpecBudget;

#define CHECK(condition) do {                                               \
    if (!(condition)) {                                                     \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n",                \
                     __FILE__, __LINE__, #condition);                       \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

static bool near(double a, double b) {
    return std::fabs(a - b) < 0.001;
}

int main() {
    // Capture and decode share one request gate. A context-over-ceiling prompt
    // reaches this predicate with zero remaining speculative budget.
    CHECK(dspark_request_can_prepare(
        true, true, false, false, 64, kDSparkMinSpecBudget));
    CHECK(!dspark_request_can_prepare(
        false, true, false, false, 64, kDSparkMinSpecBudget));
    CHECK(!dspark_request_can_prepare(
        true, false, false, false, 64, kDSparkMinSpecBudget));
    CHECK(!dspark_request_can_prepare(
        true, true, true, false, 64, kDSparkMinSpecBudget));
    CHECK(!dspark_request_can_prepare(
        true, true, false, true, 64, kDSparkMinSpecBudget));
    CHECK(!dspark_request_can_prepare(
        true, true, false, false, 0, kDSparkMinSpecBudget));
    CHECK(!dspark_request_can_prepare(
        true, true, false, false, 64, kDSparkMinSpecBudget - 1));
    CHECK(!dspark_request_can_prepare(
        true, true, false, false, 64, 0));

    // q-wide target verification is opt-in and request-local. Qualification
    // requires consecutive full exact blocks; a partial q-wide block returns
    // to exact mode from the caller's replayed frontier.
    {
        DSparkBatchVerifyGate gate(false, 48);
        for (int i = 0; i < 20; ++i) gate.note_cycle(4, 4);
        CHECK(gate.strict_cycle());
        CHECK(!gate.active());
    }
    {
        DSparkBatchVerifyGate gate(true, 0);
        CHECK(gate.active());
        gate.note_cycle(4, 3);
        CHECK(gate.strict_cycle());
        CHECK(gate.progress() == 0);
    }
    {
        DSparkBatchVerifyGate gate(true, 48);
        CHECK(gate.strict_cycle());
        for (int i = 0; i < 11; ++i) gate.note_cycle(4, 4);
        CHECK(!gate.active());
        CHECK(gate.progress() == 44);
        // Scheduler-inserted q=1 cycles neither qualify nor reset the streak.
        gate.note_cycle(1, 1);
        CHECK(gate.progress() == 44);
        gate.note_cycle(4, 4);
        CHECK(gate.active());
        CHECK(gate.progress() == 48);
        gate.note_cycle(4, 3);
        CHECK(gate.strict_cycle());
        CHECK(gate.progress() == 0);
        gate.note_cycle(4, 4);
        gate.note_cycle(4, 2);
        CHECK(gate.progress() == 0);
    }

    {
        DSparkProfitScheduler scheduler({});
        scheduler.observe_target_eval(30.0);
        for (int i = 0; i < 3; ++i) {
            CHECK(!scheduler.note_spec_cycle(3, 100.0).paused);
        }
        const auto d = scheduler.note_spec_cycle(3, 100.0);
        CHECK(!d.paused);
        CHECK(near(d.avg_accepted, 3.0));
        CHECK(near(d.extra_per_accept_ms, 70.0 / 3.0));
    }

    {
        DSparkProfitScheduler scheduler({});
        scheduler.observe_target_eval(30.0);
        for (int i = 0; i < 3; ++i) {
            CHECK(!scheduler.note_spec_cycle(1, 50.0).paused);
        }
        const auto d = scheduler.note_spec_cycle(1, 50.0);
        CHECK(d.paused);
        CHECK(d.low_accept);
        CHECK(!d.slow_accept);
        CHECK(!d.measured_unprofitable);
        CHECK(d.skip_cycles == 2);
        CHECK(scheduler.take_scheduled_skip());
        CHECK(scheduler.take_scheduled_skip());
        CHECK(!scheduler.take_scheduled_skip());
    }

    // Cross-request acceptance gate: low EWMA disables speculation, hysteresis
    // requires a materially better probe to re-enable it, and probes recur
    // periodically rather than latching off for the worker lifetime.
    {
        DSparkSchedulerConfig config;
        config.gate_min_samples = 2;
        config.gate_disable_accept_milli = 450;
        config.gate_enable_accept_milli = 700;
        config.gate_reprobe_requests = 2;
        DSparkProfitScheduler scheduler(config);
        CHECK(scheduler.allow_spec_request());
        scheduler.note_request_result(0.2f);
        CHECK(!scheduler.gate_disabled());
        scheduler.note_request_result(0.2f);
        CHECK(scheduler.gate_disabled());
        // Probes recur on the configured interval rather than latching off.
        CHECK(!scheduler.allow_spec_request());
        CHECK(scheduler.allow_spec_request());
        // First probe lifts 0.20 -> 0.55 at the probe weight: still short of
        // the 0.70 enable threshold, so one good probe cannot re-enable.
        scheduler.note_request_result(0.9f);
        CHECK(scheduler.gate_disabled());
        CHECK(scheduler.acceptance_ewma() > 0.5);
        CHECK(!scheduler.allow_spec_request());
        CHECK(scheduler.allow_spec_request());
        // Second probe crosses it. Recovery costs TWO probes, not the four or
        // five the steady-state weight would need — probes are scarce because
        // only spec-eligible requests reach the gate at all.
        scheduler.note_request_result(0.9f);
        CHECK(!scheduler.gate_disabled());
        CHECK(scheduler.acceptance_ewma() > 0.7);
        CHECK(scheduler.allow_spec_request());
    }

    // A workload whose honest acceptance sits inside [disable, enable) must
    // still recover. The enable threshold is absolute, so without a fallback
    // one transient dip disables speculation for the worker's whole life even
    // though it is profitable at 0.60.
    {
        DSparkSchedulerConfig config;
        config.gate_min_samples = 2;
        config.gate_disable_accept_milli = 450;
        config.gate_enable_accept_milli = 700;
        config.gate_reprobe_requests = 1;
        config.gate_recover_probes = 6;
        DSparkProfitScheduler scheduler(config);
        scheduler.note_request_result(0.2f);
        scheduler.note_request_result(0.2f);
        CHECK(scheduler.gate_disabled());
        // Every probe returns 0.60: profitable, comfortably above the 0.45
        // disable line, but it can never reach 0.70 no matter how many run.
        for (int i = 0; i < 8 && scheduler.gate_disabled(); ++i) {
            CHECK(scheduler.allow_spec_request());
            scheduler.note_request_result(0.6f);
        }
        CHECK(scheduler.acceptance_ewma() < 0.7);
        CHECK(!scheduler.gate_disabled());
        CHECK(scheduler.acceptance_ewma() >= 0.45);
    }

    // gate_recover_probes = 0 keeps the strict enable threshold, so a workload
    // pinned below it stays gated off. Retained as an explicit opt-in so the
    // fallback above is a policy, not an accident.
    {
        DSparkSchedulerConfig config;
        config.gate_min_samples = 2;
        config.gate_disable_accept_milli = 450;
        config.gate_enable_accept_milli = 700;
        config.gate_reprobe_requests = 1;
        config.gate_recover_probes = 0;
        DSparkProfitScheduler scheduler(config);
        scheduler.note_request_result(0.2f);
        scheduler.note_request_result(0.2f);
        CHECK(scheduler.gate_disabled());
        for (int i = 0; i < 12; ++i) {
            scheduler.allow_spec_request();
            scheduler.note_request_result(0.6f);
        }
        CHECK(scheduler.gate_disabled());
    }

    // A zero reprobe interval must not mean "never recover". Left unclamped,
    // allow_spec_request() never returns true while gated off, so no sample is
    // ever taken and the gate cannot clear for the worker's lifetime -- and it
    // was reachable from the environment.
    {
        DSparkSchedulerConfig config;
        config.gate_min_samples = 2;
        config.gate_disable_accept_milli = 450;
        config.gate_enable_accept_milli = 700;
        config.gate_reprobe_requests = 0;
        DSparkProfitScheduler scheduler(config);
        scheduler.note_request_result(0.2f);
        scheduler.note_request_result(0.2f);
        CHECK(scheduler.gate_disabled());
        bool probed = false;
        for (int i = 0; i < 4 && !probed; ++i) probed = scheduler.allow_spec_request();
        CHECK(probed);
        scheduler.note_request_result(0.9f);
        scheduler.note_request_result(0.9f);
        CHECK(!scheduler.gate_disabled());
    }

    // Recovery must not be blocked by profitability_ewma(). It is a smoothed
    // restatement of "rate >= enable" over the same samples, so gating on it as
    // well as the acceptance average was a second latch on one signal. Here
    // every probe is individually above the enable threshold, so the gate must
    // reopen as soon as the acceptance average clears it.
    {
        DSparkSchedulerConfig config;
        config.gate_min_samples = 2;
        config.gate_reprobe_requests = 1;
        DSparkProfitScheduler scheduler(config);
        scheduler.note_request_result(0.1f);
        scheduler.note_request_result(0.1f);
        CHECK(scheduler.gate_disabled());
        CHECK(scheduler.profitability_ewma() < 0.5);
        int probes = 0;
        for (int i = 0; i < 8 && scheduler.gate_disabled(); ++i) {
            if (!scheduler.allow_spec_request()) continue;
            scheduler.note_request_result(1.0f);
            ++probes;
        }
        CHECK(!scheduler.gate_disabled());
        CHECK(probes <= 3);
    }

    // A steady stream of good acceptance must never disable the gate, and the
    // steady-state weight must be the slower one (a single bad request should
    // not tip a healthy average straight through the disable threshold).
    {
        DSparkSchedulerConfig config;
        config.gate_min_samples = 2;
        DSparkProfitScheduler scheduler(config);
        for (int i = 0; i < 12; ++i) scheduler.note_request_result(0.85f);
        CHECK(!scheduler.gate_disabled());
        scheduler.note_request_result(0.0f);
        CHECK(!scheduler.gate_disabled());   // 0.85 -> ~0.64, still above 0.45
        CHECK(scheduler.acceptance_ewma() > 0.45);
    }

    // Configurable weights: alpha 1000 means "replace", so one sample decides.
    {
        DSparkSchedulerConfig config;
        config.gate_min_samples = 1;
        config.gate_alpha_milli = 1000;
        config.gate_probe_alpha_milli = 1000;
        config.gate_reprobe_requests = 1;
        DSparkProfitScheduler scheduler(config);
        scheduler.note_request_result(0.9f);
        CHECK(!scheduler.gate_disabled());
        scheduler.note_request_result(0.1f);
        CHECK(scheduler.gate_disabled());
        CHECK(scheduler.allow_spec_request());
        scheduler.note_request_result(0.9f);
        CHECK(!scheduler.gate_disabled());
    }

    {
        DSparkProfitScheduler scheduler({});
        scheduler.observe_target_eval(30.0);
        for (int i = 0; i < 3; ++i) {
            CHECK(!scheduler.note_spec_cycle(3, 130.0).paused);
        }
        const auto d = scheduler.note_spec_cycle(3, 130.0);
        CHECK(d.paused);
        CHECK(!d.low_accept);
        CHECK(d.slow_accept);
        CHECK(d.measured_unprofitable);
        CHECK(d.skip_cycles == 4);
    }

    {
        DSparkProfitScheduler scheduler({});
        scheduler.observe_target_eval(20.0);
        scheduler.observe_target_eval(40.0);
        CHECK(near(scheduler.target_eval_ms(), 25.0));
        CHECK(scheduler.tail_should_skip(9));
        CHECK(!scheduler.tail_should_skip(10));
    }

    {
        DSparkSchedulerConfig config;
        config.enabled = false;
        DSparkProfitScheduler scheduler(config);
        CHECK(!scheduler.tail_should_skip(1));
        for (int i = 0; i < 8; ++i) {
            CHECK(!scheduler.note_spec_cycle(0, 1000.0).paused);
        }
    }

    {
        DSparkSchedulerConfig config;
        config.min_avg_accepted_milli = 0;
        config.max_extra_ms_per_accept_milli = 0;
        config.max_extra_saved_ratio_milli = 0;
        DSparkProfitScheduler scheduler(config);
        scheduler.observe_target_eval(30.0);
        for (int i = 0; i < 8; ++i) {
            CHECK(!scheduler.note_spec_cycle(1, 1000.0).paused);
        }
    }

    // ── no-draft pause escalation (ds4.c:47655-47675) ───────────────────────
    // Reachable only because the confident-prefix rule can keep zero candidates.
    {
        // Cold start, never accepted, confidence at/below 0.5 -> hardest backoff.
        DSparkProfitScheduler scheduler({});
        CHECK(scheduler.lifetime_accepted() == 0);
        const uint32_t skip = scheduler.note_no_draft(true, 0.4f);
        CHECK(skip == 7);                       // cold_low_confidence_skip_cycles
        CHECK(scheduler.no_draft_cycles() == 1);
        // The pause must actually be consumed by take_scheduled_skip().
        int taken = 0;
        while (scheduler.take_scheduled_skip()) ++taken;
        CHECK(taken == 7);
    }
    {
        // Cold start but CONFIDENT -> only the base pause, not the cold branch.
        DSparkProfitScheduler scheduler({});
        CHECK(scheduler.note_no_draft(true, 0.95f) == 3);
    }
    {
        // Cold start with no confidence sample -> base pause. ds4 gates the cold
        // branch on last_confidence0_valid, so an absent sample must not escalate.
        DSparkProfitScheduler scheduler({});
        CHECK(scheduler.note_no_draft(false, 0.0f) == 3);
    }
    {
        // Has accepted, but never a long draft (>2) -> short-accept pause.
        DSparkProfitScheduler scheduler({});
        scheduler.note_spec_cycle(2, 50.0);     // 2 accepted: not "long"
        CHECK(scheduler.lifetime_accepted() == 2);
        CHECK(!scheduler.long_accept_seen());
        CHECK(scheduler.note_no_draft(true, 0.1f) == 4);
    }
    {
        // Has seen a long draft -> base pause only, even at low confidence: a
        // proven drafter should not be punished for one unconfident step.
        DSparkProfitScheduler scheduler({});
        scheduler.note_spec_cycle(3, 50.0);     // >2 -> long_accept_seen
        CHECK(scheduler.long_accept_seen());
        CHECK(scheduler.note_no_draft(true, 0.1f) == 3);
    }
    {
        // Disabled knob -> no pause at all.
        DSparkSchedulerConfig config;
        config.no_draft_skip_cycles = 0;
        DSparkProfitScheduler scheduler(config);
        CHECK(scheduler.note_no_draft(true, 0.0f) == 0);
        CHECK(!scheduler.take_scheduled_skip());
    }

    // ── target-eval baseline precedence ────────────────────────────────────
    {
        // A verify-graph q=1 sample is a COLD START only. Once a genuine AR
        // sample arrives it takes over and later fallback samples are ignored —
        // otherwise the inflated verify baseline keeps overstating savings.
        DSparkProfitScheduler scheduler({});
        scheduler.observe_target_eval_fallback(80.0);
        CHECK(near(scheduler.target_eval_ms(), 80.0));
        CHECK(!scheduler.target_eval_from_ar());

        scheduler.observe_target_eval(20.0);    // genuine AR step
        CHECK(scheduler.target_eval_from_ar());
        const double after_ar = scheduler.target_eval_ms();
        CHECK(after_ar < 80.0);                 // blended toward the AR truth

        scheduler.observe_target_eval_fallback(500.0);   // must be ignored now
        CHECK(near(scheduler.target_eval_ms(), after_ar));
    }
    {
        // Repeated AR samples converge on the AR cost.
        DSparkProfitScheduler scheduler({});
        for (int i = 0; i < 40; ++i) scheduler.observe_target_eval(20.0);
        CHECK(near(scheduler.target_eval_ms(), 20.0));
    }
    {
        // With an HONEST (cheaper) baseline the same cycle is judged
        // unprofitable. 2 accepted keeps avg above the low-accept floor of 1.5,
        // so the ratio test is the only thing deciding: extra = 100-20 = 80/cycle
        // vs saved = 20*2 = 40/cycle -> extra > saved -> pause.
        DSparkProfitScheduler scheduler({});
        scheduler.observe_target_eval(20.0);
        bool paused = false;
        for (int i = 0; i < 4; ++i) paused |= scheduler.note_spec_cycle(2, 100.0).paused;
        CHECK(paused);
    }
    {
        // Identical cycle timings and accept count, but an INFLATED baseline
        // (what the verify-graph q=1 sample gives) makes the SAME work look fine:
        // extra = 100-90 = 10/cycle vs saved = 90*2 = 180/cycle -> no pause.
        // That divergence is exactly the bias the AR baseline removes.
        DSparkProfitScheduler scheduler({});
        scheduler.observe_target_eval(90.0);
        bool paused = false;
        for (int i = 0; i < 4; ++i) paused |= scheduler.note_spec_cycle(2, 100.0).paused;
        CHECK(!paused);
    }

    // ── break_even_window (ds4.c:47702) ─────────────────────────────────────
    {
        // Default 0 = disabled: an unprofitable run must wait for the full
        // window (4 cycles) before pausing, i.e. cycles 1-3 stay unpaused.
        DSparkProfitScheduler scheduler({});
        scheduler.observe_target_eval(20.0);
        for (int i = 0; i < 3; ++i) {
            const auto d = scheduler.note_spec_cycle(2, 100.0);
            CHECK(!d.paused);
            CHECK(!d.break_even);
        }
        CHECK(scheduler.note_spec_cycle(2, 100.0).paused);
    }
    {
        // break_even_window = 2 cuts the same run at cycle 2 instead of 4, and
        // pauses for slow_skip_cycles rather than the base skip.
        DSparkSchedulerConfig config;
        config.break_even_window = 2;
        DSparkProfitScheduler scheduler(config);
        scheduler.observe_target_eval(20.0);
        CHECK(!scheduler.note_spec_cycle(2, 100.0).paused);   // cycle 1
        const auto d = scheduler.note_spec_cycle(2, 100.0);   // cycle 2
        CHECK(d.paused);
        CHECK(d.break_even);
        CHECK(d.measured_unprofitable);
        CHECK(d.skip_cycles == config.slow_skip_cycles);
        CHECK(d.saved_ms > 0.0);          // reporting fields populated
        // ds4 resets the window on a break-even pause, so the next cycle starts
        // a fresh count rather than immediately re-tripping.
        CHECK(!scheduler.note_spec_cycle(3, 21.0).paused);
    }
    {
        // A PROFITABLE run must not be cut early, however small the window.
        DSparkSchedulerConfig config;
        config.break_even_window = 1;
        DSparkProfitScheduler scheduler(config);
        scheduler.observe_target_eval(30.0);
        for (int i = 0; i < 6; ++i) {
            const auto d = scheduler.note_spec_cycle(3, 35.0);
            CHECK(!d.break_even);
        }
    }
    {
        // With no baseline yet, unprofitable() is unreachable, so break-even
        // cannot fire on an unmeasurable run.
        DSparkSchedulerConfig config;
        config.break_even_window = 1;
        DSparkProfitScheduler scheduler(config);
        for (int i = 0; i < 4; ++i) {
            CHECK(!scheduler.note_spec_cycle(3, 500.0).break_even);
        }
    }

    // ── many_no_draft (ds4.c:47733) ────────────────────────────────────────
    {
        // No-draft cycles reaching half the window is its own pause reason, and
        // escalates to slow_skip_cycles like slow_accept does.
        DSparkSchedulerConfig config;
        config.min_avg_accepted_milli = 0;      // isolate: no low-accept noise
        config.max_extra_ms_per_accept_milli = 0;
        config.max_extra_saved_ratio_milli = 0;
        DSparkProfitScheduler scheduler(config);
        scheduler.note_no_draft(true, 0.95f);
        scheduler.note_spec_cycle(0, 40.0, true);  // 1 no-draft cycle
        scheduler.note_no_draft(true, 0.95f);
        scheduler.note_spec_cycle(0, 40.0, true);  // 2
        // Drain the pause it queued so take_scheduled_skip state cannot confuse
        // the window accounting below.
        while (scheduler.take_scheduled_skip()) {}
        scheduler.note_spec_cycle(3, 40.0);
        const auto d = scheduler.note_spec_cycle(3, 40.0);   // window closes at 4
        CHECK(d.many_no_draft);                 // 2 * 2 >= 4
        CHECK(d.paused);
        CHECK(d.skip_cycles == config.slow_skip_cycles);
    }
    {
        // One no-draft in a 4-cycle window is below half -> not a reason.
        DSparkSchedulerConfig config;
        config.min_avg_accepted_milli = 0;
        config.max_extra_ms_per_accept_milli = 0;
        config.max_extra_saved_ratio_milli = 0;
        DSparkProfitScheduler scheduler(config);
        scheduler.note_no_draft(true, 0.95f);
        scheduler.note_spec_cycle(0, 40.0, true);
        while (scheduler.take_scheduled_skip()) {}
        for (int i = 0; i < 2; ++i) scheduler.note_spec_cycle(3, 40.0);
        const auto d = scheduler.note_spec_cycle(3, 40.0);
        CHECK(!d.many_no_draft);                // 1 * 2 < 4
        CHECK(!d.paused);
    }
    {
        // The counter is per-window: it must clear when the window resets, or a
        // single early no-draft would taint every later window.
        DSparkSchedulerConfig config;
        config.min_avg_accepted_milli = 0;
        config.max_extra_ms_per_accept_milli = 0;
        config.max_extra_saved_ratio_milli = 0;
        DSparkProfitScheduler scheduler(config);
        scheduler.note_no_draft(true, 0.95f);
        scheduler.note_spec_cycle(0, 40.0, true);
        scheduler.note_no_draft(true, 0.95f);
        scheduler.note_spec_cycle(0, 40.0, true);
        while (scheduler.take_scheduled_skip()) {}
        for (int i = 0; i < 2; ++i) scheduler.note_spec_cycle(3, 40.0);  // trips
        for (int i = 0; i < 3; ++i) scheduler.note_spec_cycle(3, 40.0);
        const auto d2 = scheduler.note_spec_cycle(3, 40.0);              // clean
        CHECK(!d2.many_no_draft);
        CHECK(scheduler.no_draft_cycles() == 2);   // lifetime counter persists
    }
}
