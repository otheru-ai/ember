# DSpark with DRY and temperature sampling

`DFLASH_DS4_SPEC_SAMPLING=1` enables request-aware target sampling in the serial
DSpark fresh-prompt and snapshot-restore paths. DSpark must also be configured
with `DFLASH_DS4_SPEC=1` and its draft model. The extension is off by default.
Requests retain their temperature, DRY, other penalties, top-k/top-p/min-p,
grammar masks and structural-token greedy hook. It does not change the HTTP
sampling parameters or force the request to temperature zero.

The first output token is sampled from prefill/restore logits. For each
subsequent verified row, the target sampler selects a token, records it in
request history, advances the grammar, and emits it before selecting another
row. Verification continues only while the selected tokens match the draft.
A mismatch supplies the correction; complete acceptance supplies a bonus.
Rejected draft tails never enter DRY history or consume target RNG values.
The last output remains the deferred seed for the next KV write. AR fallback
resumes with the same RNG and reconstructs history from the actual output.

This follows the target-sampling pattern in llama.cpp's
`common_sampler_sample_and_accept_n` at
`6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`. It does not use draft confidence as
a probability ratio. Acceptance can be lower under stochastic sampling, so
the existing cost-based scheduler and AR fallback remain active.

## Verification modes

- **Strict reference (default for sampled DSpark):** each conditional row uses
  the target q=1 graph, with passive DSpark feature capture. This is the
  correctness reference and is not a claim of a speedup.
- **Experimental batched mode:** requires both the existing
  `DFLASH_DS4_BATCH_VERIFY=1` and the new
  `DFLASH_DS4_SPEC_SAMPLING_BATCH_VERIFY=1`. Existing greedy batch settings
  alone cannot activate it. `DFLASH_DS4_SEQ_VERIFY=1` still forces strict mode.
  The existing warmup/profitability gate controls when a request uses q-wide
  verification. Q-wide logits can differ from AR; this mode is numerically
  approximate and requires separate quality/distribution and performance
  validation. On partial acceptance, replay restores the emitted token prefix's
  KV/compressor state without sampling again, changing emitted tokens, or
  advancing grammar/RNG a second time. It does not make those prior q-wide
  sampling decisions equivalent to q=1 decisions.

Vision requests retain their existing AR restriction. The separate resident
NPU provider path retains its existing fallback for sampled requests; this
extension does not advertise sampled resident speculation. Reply-budget
boundaries continue to hand off to AR for forced thinking closure.

## Validation and rollout

`test_sampler` includes differential AR-versus-proposed-prefix cases at
temperature 0 and 0.6 with DRY, truncation, penalties, grammar masks,
structural forcing, full/partial blocks, exact RNG-state comparison and a
stale-history negative control. Host tests cannot establish GPU logits or
KV/compressor equivalence.

Before activation, compare the exact candidate binary against AR with the
same model/draft, prompt, sampler, seed, cache state and output budget. Cover
fresh and restored prompts, EOS/cancellation, tool boundaries, thinking-budget
handoff and compression boundaries. Keep the ordinary release differential
validator as well; its greedy check alone cannot certify this sampled path.
Run the strict reference before the experimental batched mode. Record
per-request actual speculation, offered/accepted candidates, q widths,
sampler/readback/replay time, tokens, finish reason and time to a valid tool
completion. Compare distributions under the project's accepted numerical
criteria and label approximate evidence; never infer parity from matching
argmax alone.

For performance compare AR+DRY and sampled DSpark+DRY at identical temperature
and with alternated run order. The startup line `[ds4-spec-sampling]` records
whether q-wide verification was requested/strict mode forced; existing
`[ds4-step]` diagnostics record actual cycle widths when
`DFLASH_DS4_SPEC_STEP_LOG=1`. Request results retain `spec_decode_ran`,
`spec_cycles` and acceptance metrics. Activation requires independently
reviewed correctness and a repeatable end-to-end improvement; no speedup is
implied by compiling or passing host tests.

Do not change a running fixed-design study to test this feature. Disabling
`DFLASH_DS4_SPEC_SAMPLING` restores the existing greedy-only DSpark eligibility
and routes DRY/temperature requests through AR.
