// Rolling profitability policy for DSpark speculative decoding.
//
// Kept independent of the GPU runtime so the policy can be unit-tested in the
// normal CPU-only build. Timing inputs are wall-clock milliseconds.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace dflash::common {

// Minimum target-token runway needed for feature capture and speculative
// decode to repay their setup cost. Callers pass a budget already clamped by
// both the reply allowance and the context ceiling.
constexpr int kDSparkMinSpecBudget = 16;

// One eligibility predicate for both prefill capture and decode. Capture is a
// passive output of the ordinary target graph, but exact tokenwise prefill and
// feature readback still have a cost. Prepare them only when the request can
// actually enter DSpark, rather than merely because a drafter is loaded.
inline bool dspark_request_can_prepare(bool spec_enabled,
                                       bool have_drafter,
                                       bool force_ar_decode,
                                       bool sampling_requires_ar,
                                       int n_gen,
                                       int spec_budget) {
    return spec_enabled && have_drafter && !force_ar_decode &&
           !sampling_requires_ar && n_gen > 0 &&
           spec_budget >= kDSparkMinSpecBudget;
}

// Per-request qualification gate for the numerically different q-wide target
// graph. Exact q=1 verification must first observe `warmup_tokens` worth of
// consecutive fully accepted blocks. A partial batched block is replayed by
// the caller and sends the request back through qualification from that exact
// frontier. Keeping this policy GPU-free makes its safety transitions directly
// unit-testable.
class DSparkBatchVerifyGate {
public:
    DSparkBatchVerifyGate(bool requested, uint32_t warmup_tokens)
        : requested_(requested), warmup_tokens_(warmup_tokens),
          active_(requested && warmup_tokens == 0) {}

    bool active() const { return active_; }
    bool strict_cycle() const { return !active_; }
    uint32_t progress() const { return progress_; }

    void note_cycle(int offered_tokens, int accepted_tokens) {
        if (!requested_ || offered_tokens < 2 || accepted_tokens < 1) return;
        if (active_) {
            if (accepted_tokens < offered_tokens) {
                active_ = false;
                progress_ = 0;
            }
            return;
        }
        if (accepted_tokens != offered_tokens) {
            progress_ = 0;
            return;
        }
        progress_ += (uint32_t) accepted_tokens;
        if (progress_ >= warmup_tokens_) active_ = true;
    }

private:
    bool requested_ = false;
    uint32_t warmup_tokens_ = 0;
    bool active_ = false;
    uint32_t progress_ = 0;
};

struct DSparkSchedulerConfig {
    bool enabled = true;
    uint32_t window = 4;
    uint32_t skip_cycles = 2;
    uint32_t slow_skip_cycles = 4;
    uint32_t min_avg_accepted_milli = 1500;
    uint32_t max_extra_ms_per_accept_milli = 28000;
    uint32_t max_extra_saved_ratio_milli = 1000;
    uint32_t tail_min_tokens = 10;
    // ── no-draft pause (ds4.c:47655-47675) ──────────────────────────────────
    // Reachable only now that the confident-prefix rule can keep zero
    // candidates. Defaults are ds4's:
    //   DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP                 3
    //   DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP    4
    //   DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP      7
    //   DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI   500  (0.5 survival)
    uint32_t no_draft_skip_cycles = 3;
    uint32_t short_accept_no_draft_skip_cycles = 4;
    uint32_t cold_low_confidence_skip_cycles = 7;
    uint32_t cold_low_confidence_milli = 500;
    // ── early break-even check (ds4.c:47702) ────────────────────────────────
    // Lets the unprofitability test fire BEFORE the full `window` has elapsed:
    // once at least this many cycles are in, a clearly-losing run is cut after
    // break_even_window cycles instead of waiting for window (4). ds4 pauses for
    // slow_skip_cycles and resets the window when it trips.
    //
    // 0 disables it, which is ds4's own default
    // (DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW). Kept off here for the same
    // reason: it trades a faster reaction for a noisier verdict, since a 1-2
    // cycle sample of extra_ms/saved_ms is easily dominated by one slow step.
    uint32_t break_even_window = 0;
    // Cross-request acceptance gate. Hysteresis prevents one hard passage
    // from permanently disabling speculation; while gated off, one request
    // every `gate_reprobe_requests` is allowed to re-measure the content.
    //
    // IMPORTANT: these counters advance in *spec-eligible* requests, not wall
    // requests. Only requests that could actually speculate reach the gate —
    // sampled and forced-AR ones short-circuit before it. In an agent workload
    // the great majority of turns are tool-result continuations carrying
    // force_ar_decode, so eligible requests are a small minority of traffic and
    // every count here is worth many times its face value in wall requests.
    // Keep the reprobe interval small for that reason.
    uint32_t gate_disable_accept_milli = 450;  // disable below 0.45
    uint32_t gate_enable_accept_milli = 700;   // re-enable at 0.70
    uint32_t gate_min_samples = 4;
    uint32_t gate_reprobe_requests = 3;
    // Samples taken while gated off are scarce and deliberately paid for, so
    // they adapt faster than the steady-state stream: recovery must not need
    // three or four probes to climb out of the disable band. A wrong re-enable
    // is cheap — the disable path re-fires at the normal weight.
    uint32_t gate_alpha_milli = 250;        // steady state
    uint32_t gate_probe_alpha_milli = 500;  // while gated off
    // Recovery must not depend on a bar the workload can never clear.
    //
    // `gate_enable_accept_milli` is an absolute threshold, so if a workload's
    // honest steady-state acceptance sits inside the [disable, enable) band —
    // profitable, but never 0.70 — one transient dip disables speculation for
    // the life of the worker. The 0.70 bar happens to be reachable for the
    // current agent workload (161 production samples: mean 0.848, 93% >= 0.70),
    // but nothing in the design guarantees that for a different draft model or
    // context regime, and the failure is silent.
    //
    // So after this many probes with the gate still off, recovery falls back to
    // the disable threshold: the question becomes "is speculation still
    // unprofitable?" rather than "is it excellent?". Set 0 to require the full
    // enable threshold forever (the old, latch-prone behaviour).
    uint32_t gate_recover_probes = 6;
};

struct DSparkSchedulerDecision {
    bool paused = false;
    bool low_accept = false;
    bool slow_accept = false;
    bool measured_unprofitable = false;
    // Fired before the full window elapsed (ds4's break-even pause).
    bool break_even = false;
    // No-draft cycles reached half the window (ds4's many_no_draft).
    bool many_no_draft = false;
    uint32_t skip_cycles = 0;
    double avg_accepted = 0.0;
    double extra_per_accept_ms = 0.0;
    double saved_ms = 0.0;
    double extra_ms = 0.0;
};

class DSparkProfitScheduler {
public:
    explicit DSparkProfitScheduler(DSparkSchedulerConfig config)
        : config_(config) {
        if (config_.window == 0) config_.window = 4;
        // A zero reprobe interval means allow_spec_request() never lets a
        // request through while gated off, so note_request_result() is never
        // called, so gate_disabled_ can never clear: speculation stays off for
        // the life of the worker thread with no way back. That is a latch, not
        // a configuration, and it was reachable from the environment via
        // DFLASH_DS4_SPEC_GATE_REPROBE_REQUESTS=0. Clamp it the same way
        // `window` is clamped above. To turn speculation off, set
        // DFLASH_DS4_SPEC=0.
        if (config_.gate_reprobe_requests == 0) config_.gate_reprobe_requests = 1;
    }

    bool enabled() const { return config_.enabled; }

    // Persistent worker-scoped profitability gate. It is deliberately
    // request-level: short generations cannot fill a fresh per-call window.
    bool allow_spec_request() {
        if (!config_.enabled || !gate_disabled_) return true;
        if (config_.gate_reprobe_requests == 0) return false;
        if (++gate_requests_since_probe_ < config_.gate_reprobe_requests) return false;
        gate_requests_since_probe_ = 0;
        return true;
    }

    // Only ever called for a request that actually speculated (the caller
    // guards on spec_ran). Feeding a gated-off request's 0.0 in here would
    // drive the average further below the disable threshold and latch the gate
    // off permanently, which is precisely what the hysteresis exists to avoid.
    void note_request_result(float accept_rate) {
        if (!std::isfinite(accept_rate) || accept_rate < 0.0f) return;
        const double rate = std::min(1.0, (double) accept_rate);
        const double enable = config_.gate_enable_accept_milli / 1000.0;
        // A sample taken while gated off is a probe: weight it harder, because
        // probes are rare by construction and recovery is measured in them.
        const uint32_t alpha_milli = gate_disabled_
            ? config_.gate_probe_alpha_milli : config_.gate_alpha_milli;
        const double alpha = std::min(1.0, alpha_milli / 1000.0);
        if (gate_samples_ == 0) {
            gate_accept_ewma_ = rate;
            gate_profit_ewma_ = rate >= enable ? 1.0 : 0.0;
        } else {
            gate_accept_ewma_ = (1.0 - alpha) * gate_accept_ewma_ + alpha * rate;
            const double profitable = rate >= enable ? 1.0 : 0.0;
            gate_profit_ewma_ =
                (1.0 - alpha) * gate_profit_ewma_ + alpha * profitable;
        }
        ++gate_samples_;
        if (gate_samples_ < config_.gate_min_samples) return;
        const double disable = config_.gate_disable_accept_milli / 1000.0;
        if (gate_disabled_) ++gate_probes_since_disable_;
        // After gate_recover_probes fruitless probes, stop asking for
        // excellence and ask only whether speculation is still unprofitable.
        // Without this, an absolute enable threshold the workload cannot reach
        // makes the disable permanent.
        const bool recovery_stalled =
            config_.gate_recover_probes != 0 &&
            gate_probes_since_disable_ >= config_.gate_recover_probes;
        const double recover_bar = recovery_stalled ? disable : enable;
        if (!gate_disabled_ && gate_accept_ewma_ < disable) {
            gate_disabled_ = true;
            gate_requests_since_probe_ = 0;
            gate_probes_since_disable_ = 0;
        } else if (gate_disabled_ && gate_accept_ewma_ >= recover_bar) {
            // Re-enable on the acceptance average alone. gate_profit_ewma_ is
            // a smoothed restatement of "rate >= enable" over the same samples,
            // so requiring it too was a second latch on one signal: it delayed
            // recovery by extra probes without adding information the
            // 0.45/0.70 hysteresis band does not already provide. It stays as
            // telemetry via profitability_ewma().
            gate_disabled_ = false;
            gate_requests_since_probe_ = 0;
            gate_probes_since_disable_ = 0;
        }
    }

    bool gate_disabled() const { return gate_disabled_; }
    double acceptance_ewma() const { return gate_accept_ewma_; }
    double profitability_ewma() const { return gate_profit_ewma_; }
    uint32_t gate_samples() const { return gate_samples_; }

    bool tail_should_skip(int remaining_tokens) const {
        return config_.enabled && config_.tail_min_tokens != 0 &&
               remaining_tokens >= 0 &&
               (uint64_t) remaining_tokens < config_.tail_min_tokens;
    }

    bool take_scheduled_skip() {
        if (!config_.enabled || skip_remaining_ == 0) return false;
        --skip_remaining_;
        ++skipped_cycles_;
        return true;
    }

    // ds4's no-draft pause, with its escalation ladder (ds4.c:47655-47675).
    //
    // Called when the drafter contributed no usable candidate — in ember that
    // means the confident-prefix rule kept zero. The pause length escalates:
    //   - base                                        : no_draft_skip_cycles (3)
    //   - has accepted before, but never a long draft : short_accept (4)
    //   - has NEVER accepted and confidence is cold   : cold_low_confidence (7)
    // The last case is the important one: a drafter that is both unproven and
    // unconfident should back off hard rather than retry every step.
    //
    // `confidence0_valid` mirrors ds4's s->dspark_last_confidence0_valid — with
    // no confidence sample we do not take the cold branch, exactly as ds4 does.
    uint32_t note_no_draft(bool confidence0_valid, float confidence0) {
        if (!config_.enabled || config_.no_draft_skip_cycles == 0) return 0;
        uint32_t skip = config_.no_draft_skip_cycles;
        if (lifetime_accepted_ != 0 && !long_accept_seen_) {
            skip = std::max(skip, config_.short_accept_no_draft_skip_cycles);
        } else if (lifetime_accepted_ == 0 && confidence0_valid &&
                   confidence0 * 1000.0f <=
                       (float) config_.cold_low_confidence_milli) {
            skip = std::max(skip, config_.cold_low_confidence_skip_cycles);
        }
        if (skip_remaining_ < skip) skip_remaining_ = skip;
        ++no_draft_cycles_;
        return skip;
    }

    // Feed a genuine single-token TARGET decode time.
    //
    // ds4 samples this around its ordinary target decode (ds4.c:59514), NOT
    // around a verify-graph step. That distinction matters: the verify path
    // carries fixed overhead a plain AR step never pays, so a baseline taken
    // there is inflated — and since saved_ms = baseline x accepted while
    // extra_ms = elapsed - baseline, an inflated baseline simultaneously
    // OVERSTATES savings and UNDERSTATES overhead. Both errors push the
    // profitability verdict toward "keep speculating".
    void observe_target_eval(double elapsed_ms) {
        if (elapsed_ms <= 0.0 || !std::isfinite(elapsed_ms)) return;
        target_eval_from_ar_ = true;
        blend_target_eval(elapsed_ms);
    }

    // Verify-graph q=1 sample: a seed-only cycle that still traverses the verify
    // machinery. Used ONLY until a genuine AR sample arrives, because it is
    // biased high for the reasons in observe_target_eval. Keeping it as a cold
    // start beats having no baseline at all — with target_eval_ms == 0 both
    // slow_accept and measured_unprofitable are unreachable, leaving only the
    // weak low_accept test, which is how spec ran at -45% without pausing.
    void observe_target_eval_fallback(double elapsed_ms) {
        if (elapsed_ms <= 0.0 || !std::isfinite(elapsed_ms) ||
            target_eval_from_ar_) return;
        blend_target_eval(elapsed_ms);
    }

    bool target_eval_from_ar() const { return target_eval_from_ar_; }

    DSparkSchedulerDecision note_spec_cycle(uint32_t accepted_drafts,
                                            double elapsed_ms,
                                            bool no_draft = false) {
        DSparkSchedulerDecision decision;
        if (!config_.enabled) return decision;

        ++cycles_;
        accepted_ += accepted_drafts;
        if (no_draft) ++window_no_draft_;
        // Lifetime counters drive the no-draft escalation above. ds4 keeps the
        // equivalents on the session (dspark_sched_lifetime_accepted,
        // dspark_sched_long_accept_seen) so they survive across decode calls;
        // this instance is worker-scoped for the same reason.
        lifetime_accepted_ += accepted_drafts;
        if (accepted_drafts > 2u) long_accept_seen_ = true;

        // Ember's verifier cycle includes the seed evaluation that plain AR
        // would have paid anyway. Dwarfstar measures only cost beyond that
        // target eval, so subtract the rolling q=1 baseline before comparing
        // speculative overhead with AR work saved.
        if (target_eval_ms_ > 0.0 && elapsed_ms > 0.0 &&
            std::isfinite(elapsed_ms)) {
            extra_ms_ += std::max(0.0, elapsed_ms - target_eval_ms_);
            saved_ms_ += target_eval_ms_ * (double) accepted_drafts;
            timed_accepted_ += accepted_drafts;
        }

        // Early break-even check (ds4.c:47702). Evaluated BEFORE the full-window
        // gate so a clearly-losing run is cut after break_even_window cycles
        // rather than waiting for `window`. ds4 pauses for slow_skip_cycles and
        // resets the window; it returns immediately, so the regular tests below
        // do not also run on this cycle. Disabled at 0 (ds4's default).
        if (config_.break_even_window != 0 &&
            cycles_ >= config_.break_even_window && unprofitable()) {
            decision.paused = true;
            decision.break_even = true;
            decision.measured_unprofitable = true;
            decision.skip_cycles = config_.slow_skip_cycles;
            decision.avg_accepted = cycles_ != 0
                ? (double) accepted_ / (double) cycles_ : 0.0;
            decision.extra_ms = extra_ms_;
            decision.saved_ms = saved_ms_;
            decision.extra_per_accept_ms = timed_accepted_ != 0
                ? extra_ms_ / (double) timed_accepted_ : 0.0;
            skip_remaining_ = decision.skip_cycles;
            reset_window();
            return decision;
        }

        if (cycles_ < config_.window) return decision;

        decision.avg_accepted = cycles_ != 0
            ? (double) accepted_ / (double) cycles_ : 0.0;
        decision.extra_ms = extra_ms_;
        decision.saved_ms = saved_ms_;
        decision.extra_per_accept_ms = timed_accepted_ != 0
            ? extra_ms_ / (double) timed_accepted_ : 0.0;
        decision.low_accept =
            accepted_ * 1000ull <
            (uint64_t) config_.min_avg_accepted_milli * cycles_;
        decision.slow_accept =
            config_.max_extra_ms_per_accept_milli != 0 &&
            target_eval_ms_ > 0.0 && timed_accepted_ != 0 &&
            decision.extra_per_accept_ms * 1000.0 >
                (double) config_.max_extra_ms_per_accept_milli;
        decision.measured_unprofitable = unprofitable();
        // ds4's many_no_draft (ds4.c:47733): no-draft cycles reaching half the
        // window is itself a pause reason, escalated like slow_accept.
        //
        // Confidence-rejected q=1 steps call note_spec_cycle(..., true), so the
        // denominator includes no-draft cycles exactly as Dwarfstar does.
        decision.many_no_draft =
            window_no_draft_ != 0 && window_no_draft_ * 2u >= cycles_;

        if (decision.low_accept || decision.slow_accept ||
            decision.measured_unprofitable || decision.many_no_draft) {
            decision.paused = true;
            decision.skip_cycles = config_.skip_cycles;
            if (decision.slow_accept || decision.measured_unprofitable ||
                decision.many_no_draft) {
                decision.skip_cycles =
                    std::max(decision.skip_cycles, config_.slow_skip_cycles);
            }
            skip_remaining_ = decision.skip_cycles;
        }
        reset_window();
        return decision;
    }

    double target_eval_ms() const { return target_eval_ms_; }
    uint64_t skipped_cycles() const { return skipped_cycles_; }
    uint64_t no_draft_cycles() const { return no_draft_cycles_; }
    uint64_t lifetime_accepted() const { return lifetime_accepted_; }
    bool long_accept_seen() const { return long_accept_seen_; }

private:
    // Single definition of "measured unprofitable", shared by the early
    // break-even check and the full-window check so they cannot diverge.
    bool unprofitable() const {
        return config_.max_extra_saved_ratio_milli != 0 &&
               target_eval_ms_ > 0.0 && timed_accepted_ != 0 &&
               saved_ms_ > 0.0 &&
               extra_ms_ * 1000.0 >
                   saved_ms_ * (double) config_.max_extra_saved_ratio_milli;
    }

    void blend_target_eval(double elapsed_ms) {
        if (target_eval_ms_ <= 0.0) {
            target_eval_ms_ = elapsed_ms;
        } else {
            // Smooth isolated samples without making the scheduler slow to
            // follow context-dependent target cost.
            target_eval_ms_ = 0.75 * target_eval_ms_ + 0.25 * elapsed_ms;
        }
    }

    void reset_window() {
        cycles_ = 0;
        accepted_ = 0;
        window_no_draft_ = 0;
        timed_accepted_ = 0;
        extra_ms_ = 0.0;
        saved_ms_ = 0.0;
    }

    DSparkSchedulerConfig config_;
    uint32_t cycles_ = 0;
    uint64_t accepted_ = 0;
    uint64_t timed_accepted_ = 0;
    uint32_t skip_remaining_ = 0;
    uint64_t skipped_cycles_ = 0;
    double extra_ms_ = 0.0;
    double saved_ms_ = 0.0;
    double target_eval_ms_ = 0.0;
    // Lifetime (not per-window) state for the no-draft escalation.
    uint64_t lifetime_accepted_ = 0;
    uint64_t no_draft_cycles_ = 0;
    uint32_t window_no_draft_ = 0;
    bool long_accept_seen_ = false;
    bool target_eval_from_ar_ = false;
    bool gate_disabled_ = false;
    uint32_t gate_samples_ = 0;
    uint32_t gate_requests_since_probe_ = 0;
    uint32_t gate_probes_since_disable_ = 0;
    double gate_accept_ewma_ = 0.0;
    double gate_profit_ewma_ = 0.0;
};

// Worker-scoped scheduler instance.
//
// ds4 stores this state on the session (s->dspark_sched_*), so it persists
// across decode calls: the 4-cycle profitability window, the rolling target-eval
// baseline, and the lifetime accept counters all carry forward. Ember's spec
// entry point is per-request, so a per-call scheduler would restart the window
// and lose the baseline on every request — short generations would never reach
// the window at all, and target_eval_ms would always start cold. Spec decode
// runs only on the single generation worker, so thread_local gives the same
// lifetime without locking.
//
// `config` is applied on first use for this thread; later calls reuse it.
DSparkProfitScheduler & dspark_worker_scheduler(const DSparkSchedulerConfig & config);

// Same worker instance, with the environment parsed once per thread. The
// scheduler is constructed on first use and ignores the config argument on
// every later call, so re-reading ~12 environment variables per call only to
// discard the result is pure waste. Request gating runs several times per
// request; use this form there.
DSparkProfitScheduler & dspark_worker_scheduler();

// Resolve the engine's scheduler environment once at worker startup. Kept
// public so backend request gating uses the exact same configuration as the
// speculative loop rather than silently falling back to defaults.
DSparkSchedulerConfig dspark_scheduler_config_from_env();

// Feed a genuine single-token target decode time into the worker's scheduler.
// Safe to call before any spec decode has run (it is then a no-op).
void dspark_worker_note_target_eval(double elapsed_ms);

}  // namespace dflash::common
