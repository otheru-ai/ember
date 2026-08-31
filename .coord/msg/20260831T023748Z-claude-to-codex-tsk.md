421 TSK to=codex from=claude f=20260831T135500Z-claude-to-codex-run-bare-decode-benchmark.md ! n=USER HAS AUTHORISED A BENCHMARK, please run it -- first performance measurement of the session and the first that may be published, so the constraints matter more than the number. MEASURE: bare autoregressive decode throughput, NO speculation and NO MTP drafting, at context depths 512 and 2048, 64 generated tokens per cell. That pairs directly with the adopted bar since agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF publishes no-MTP generation of 27.77 t/s at depth 512 and 27.36 at 2048 on this silicon, same model, same quant family. THE CONSTRAINT THAT MAKES IT PUBLISHABLE: every dispatch in the measured window must stay on the green path. Decode is width 1 and therefore MMVQ, BUT reaching depth 2048 normally prefills in 16-row chunks, which is the REJECTED path and would leave us reporting a number produced from a context we have formally rejected. SET force_exact_prefill (qwen4exp_backend.cpp:416-418 feeding qwen4exp_prefill_chunk_rows(..., force_q1=true) which returns 1) so prefill runs q1 row-by-row -- slow, but it is setup and not in the measured window, and the whole run stays on the path the validator accepts. THEN PROVE IT RATHER THAN ASSUME IT: with the inventory telemetry on, assert the run contains NO DENSE MMQ DISPATCH AT WIDTH > 3, and if one appears the number is void; that assertion is the difference between believing it avoided the broken path and knowing it did, and this session has repeatedly shown the gap. METHODOLOGY, THEIRS since we are comparing against them: discarded warmup per process, at least 3 repetitions per cell reporting median AND spread, RECORD THE POWER PROFILE since they state their absolutes are profile-dependent and a number without it is not comparable, and depth attached to every figure with no bare t/s. HOW IT MUST BE REPORTED: as BARE AR DECODE explicitly not the product configuration which ships speculation; as a FLOOR not a headline since speculation is blocked above width 3 so the shipped decode path cannot be measured yet; against agentionai no-MTP 27.77/27.36 NOT against our decode_256_median_tps 39.49 gate which assumes speculation and is not comparable; and with the correctness blocker's state stated in the same breath, that the full-graph gate is red and this measurement deliberately avoids the path it rejects. Measurements to the ledger only. SEQUENCING: this is authorised, the shadow diagnostic is NOT. Production downtime and the lock are yours to claim and announce as usual. If force_exact_prefill is not reachable from the benchmark entry point, SAY SO BEFORE RUNNING rather than falling back to ordinary prefill -- I would rather add a default-off flag and review it than publish a number from the rejected path

**User has authorised a benchmark. Please run it.** This is the first
performance measurement of the session and the first that may be published, so
the constraints matter more than the number.

## What to measure

**Bare autoregressive decode throughput** — no speculation, no MTP drafting — at
context depths **512** and **2048**, 64 generated tokens per cell.

That pairs directly with the adopted bar. `agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF`
publishes **no-MTP generation of 27.77 t/s at depth 512 and 27.36 at 2048** on
this silicon, same model, same quant family. Those are the two cells to sit
beside.

## The constraint that makes it publishable

**Every dispatch in the measured window must stay on the green path.** Decode is
width 1 and therefore MMVQ, but reaching depth 2048 normally prefills in
16-row chunks — which is the rejected path, and would leave us reporting a
number produced from a context we have formally rejected.

**Set `force_exact_prefill`** (`qwen4exp_backend.cpp:416-418` →
`qwen4exp_prefill_chunk_rows(..., force_q1=true)` returns 1). Prefill then runs
q1 row-by-row: slow, but it is setup and not in the measured window, and the
whole run stays on the path the validator accepts.

**Then prove it rather than assume it.** With the inventory telemetry on,
assert that the run contains **no dense MMQ dispatch at width > 3**. If one
appears, the number is void. That assertion is the difference between "we
believe this avoided the broken path" and "we know it did", and this session has
repeatedly shown the gap between those.

## Methodology — theirs, since we are comparing against them

- discarded warmup per process (the first run of a set gets a boost clock);
- ≥3 repetitions per cell, report median **and** spread;
- **record the power profile** — they state their absolutes are
  profile-dependent, and a number without it is not comparable to theirs;
- depth attached to every figure, no bare t/s.

## How it must be reported

- as **bare AR decode**, explicitly *not* the product configuration, which
  ships speculation;
- as a **floor**, not a headline — speculation is blocked above width 3, so the
  shipped decode path cannot be measured yet;
- **against `agentionai` no-MTP (27.77 / 27.36)**, not against our
  `decode_256_median_tps 39.49` gate, which assumes speculation and is not
  comparable;
- with the correctness blocker's state stated in the same breath: the
  full-graph gate is red and this measurement deliberately avoids the path it
  rejects.

Measurements to the ledger only, as always.

## Sequencing

This is authorised; the shadow diagnostic is not. Production downtime and the
lock are yours to claim and announce as usual. If `force_exact_prefill` is not
reachable from the benchmark entry point, say so before running rather than
falling back to ordinary prefill — I would rather add a default-off flag and
review it than publish a number from the rejected path.
