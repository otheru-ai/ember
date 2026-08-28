# Qwen3.8 Flash Next quant and intervention bakeoff

This is an experiment protocol, not a selected release recipe. A recipe is
confirmed only after the scripts accept measured correctness, quality,
throughput, artifact-size, RSS, GTT, and accounted UMA evidence from `otheru`.
Estimated values and missing arms fail closed.

## Builder limits and bounded conversion

The certification host reports `MemTotal: 131150288 kB`, exactly
134,297,894,912 bytes (about 125.07 GiB). Its root filesystem has about 1.487
TiB free; `/srv/models` has only about 295.4 GiB free. The pinned source weight
inventory is 360,000,192,888 bytes, so both the snapshot and conversion
workspace belong on the root filesystem.

The live gfx1151 mapping ceiling is slightly smaller than physical memory:
`/sys/module/ttm/parameters/pages_limit` is 32,505,856 pages, exactly
133,143,986,176 bytes (124.0 GiB), and `rocminfo` reports the same 124.0 GiB
global pool. Certification therefore rejects a measured GTT peak above that
cap even when total UMA remains below `MemTotal`.

The pinned external `kingjones777` ROCmFP4 STRIX_LEAN comparison artifact
(`dec9c5c1053ef814cfaa39b342efd4cdd721ef0b`) was downloaded as three
hash-verified shards totaling 105,753,530,752 bytes. It is a loader and speed
control, not a viable release recipe for this host: artifact plus the required
34,359,738,368-byte runtime reserve is 140,113,269,120 bytes, already
5,815,374,208 bytes above the measured `MemTotal` before MTP or vision is
enabled. No runtime or throughput claim follows from this inventory check.

Ordinary conversion is not viable on this host: its approximately 204 GB
intermediate representation exceeds RAM. The first supposedly bounded
`--use-temp-file` control run was also not viable: run `33050472929` was
cgroup-OOM-killed after reading 300,364,726,272 of 360,000,192,888 source
bytes, at a measured peak RSS of 127,269,264 KiB. The PLE converter flushed
only after populating its entire 204.8 GB file-backed F32 map, so dirty mmap
pages accumulated as cgroup resident memory.

The pinned llama.cpp base is now combined with the digest-pinned
`qwen4exp-ple-cgroup-writeback.patch`. It flushes and `MADV_DONTNEED`s each
completed PLE shard and marks the final clean read sequential. A second control
run (`33058038825`) proved that this removed dirty-map pressure, but then exposed
another monolithic allocation: llama.cpp's grouped BF16 converter tried to
materialize the 204.8 GB F32 PLE table as a roughly 102.4 GB anonymous array and
was killed at 128,524,508 KiB RSS. The pinned patch therefore also keeps only
the transient PLE tensor F32 when `--use-temp-file` is active. The release
quantizer still applies and verifies the selected PLE override, so the F32
intermediate is a storage mechanism rather than a release precision choice.
The converter patch is part of the tool provenance. A third control run
(`33077679060`) completed conversion, then exposed a separate upstream splitter
allocation: `llama-gguf-split` resized its copy buffer to the full 204.8 GB PLE
tensor and aborted with `std::bad_alloc` inside the 125 GiB cgroup. Ember's
second digest-pinned patch, `gguf-split-bounded-copy.patch`, retains the exact
split format while copying every tensor through one fixed 16 MiB buffer.
llama.cpp may differ from its pinned commit in those two exact files only.
`--use-temp-file` cannot split while converting, so Ember performs this private
lifecycle:

1. stream safetensors into one private mostly-BF16 GGUF with the PLE staging
   tensor retained F32, spilling through `TMPDIR` inside the transaction
   directory;
2. use `llama-gguf-split` from the same pinned llama.cpp checkout to create 48G
   shards;
3. verify the complete ordered GGUF set and remove the unsplit BF16 file;
4. preflight and quantize the split set with `--keep-split`;
5. atomically commit the verified directory without overwriting an existing
   result.

The input snapshot, temporary payload, unsplit output, and split output can
coexist at different points, so the measured-path floor is 1152 GiB free. The
120 GiB physical-RAM floor merely admits the patched route; it is not evidence
that conversion succeeded within memory. The build record retains the failed
run measurements and leaves patched peak RSS and wall time pending until the
target rerun measures them.

```sh
git -C /root/qwen-work/llama.cpp apply \
  /root/qwen-work/ember/patches/llama.cpp/qwen4exp-ple-cgroup-writeback.patch
git -C /root/qwen-work/llama.cpp apply \
  /root/qwen-work/ember/patches/llama.cpp/gguf-split-bounded-copy.patch
python3 scripts/qwen_quantize.py \
  --snapshot-dir /root/qwen-work/snapshot \
  --snapshot-revision f5d08274bafd880402bd16f5e3e6c514136ec06c \
  --stock-control \
  --llama-cpp-dir /root/qwen-work/llama.cpp \
  --gguf-splitter /root/qwen-work/llama.cpp/build/bin/llama-gguf-split \
  --bounded-memory-temp \
  --conversion-memory-limit-bytes 134217728000 \
  --rocmfpx-dir /root/qwen-work/ROCmFPX \
  --ember-dir /root/qwen-work/ember \
  --ember-revision "$(git rev-parse HEAD)" \
  --quantizer /root/qwen-work/ember/build-rocm/ember-gguf-quantize \
  --work-dir /root/qwen-work/stock-rocmi4 \
  --execute
```

The measured command must itself run in a fresh cgroup-v2 container created
with `--memory 125g --memory-swap 125g`. The pipeline requires the resulting
`memory.max` to equal 134,217,728,000 bytes, requires `memory.swap.max` to be
zero, and binds the observed cgroup peak, converter child RSS, and wall time
into the build record. The workflow also hashes the outer `/usr/bin/time`
evidence together with that record.

The stock control carries explicit `none_control` and
`control_only_requires_manifest_for_release` metadata on every shard and is
always final-ineligible. Those labels are negative control evidence, not proof
of a weight intervention. It exists to establish correctness, quality, and
performance and to capture the 48×2560 per-prompt residual-writer outputs. The
capture point is the output of each GDN ``ssm_out`` or QSA ``attn_output``
projection, before hyper-connection injection, so every direction inhabits the
same 2560-row space as the weight matrix it modifies. A
release package still requires a separately measured intervention artifact.

## Pinned and disjoint corpora

The adapter accepts only OtherU quant-pipeline revision
`a3c6a728510f91394e991504951ac316cd3a89af` on branch
`ember-contract-and-drafter-fix`, and verifies each raw input digest before
reading it.

```sh
python3 scripts/qwen_corpus_adapter.py \
  --source-dir /root/qwen-work/otheru-quant-pipeline \
  --output-dir /root/qwen-work/qwen-corpora
```

At the pinned revision the deterministic result is:

- 32 harmless and 32 harmful direction-extraction rows;
- 134 sweep-validation rows;
- 134 final-heldout rows;
- zero canonical-message overlap among all four outputs.

Within each evaluation suite, rows are sorted by SHA-256 of canonical chat
messages and divided equally. The final half cannot select a recipe. The
upstream `overtrigger.json` is digest-pinned but deliberately excluded because
it contains environment-derived private material; a separately reviewed,
sanitized tool-use suite is required before tool-use quality can pass.

## Bakeoff and measured decision

Generate the expanded 4-lambda × 4-layer-policy plan:

```sh
python3 scripts/qwen_bakeoff.py \
  --corpus-dir /root/qwen-work/qwen-corpora \
  --output /root/qwen-work/qwen-bakeoff-plan.json
```

The sweep uses positive projection-removal strengths 0.25, 0.5, 0.75, and
1.0 over layers 10–42, upper 24, upper 12, and the non-QSA subset of layers
10–42. The 10–42 band is an exploratory transfer hypothesis from the pinned
OtherU DeepSeek result, where editing early layers broke coherence; it is not a
proven Qwen policy. All four policies must pass Qwen's held-out quality gates.
The sweep first
selects an intervention configuration using only sweep-validation. That fixed
configuration then runs a six-arm exact-runtime cross-pair: each of the
ROCMI4+Q6_K, routed-expert ROCmFP4 FAST+Q6_K, and broad-matrix ROCmFP4
FAST+Q6_K main artifacts is measured with both the homogeneous ROCMI4 and
homogeneous ROCmFP4 FAST MTP companions at depth 3. Main format and MTP format
are independent experiment variables; the release profile's per-arm MTP field
is only the companion build default.

After the winning main/MTP pair is externally attested, the same exact main
and companion inventories are reused to sweep MTP depths 1, 2, 3, and 4. The
assessment, artifact/runtime identity, phase ledger, and final confirmation
all bind both MTP matrix contract and depth. Only the sealed depth winner can
unlock final-heldout. ROCMI4 W4A4 remains a separately inventoried auxiliary
performance control: it is lossy, final-ineligible, and never participates in
the exact-runtime winner ledger. The stock model is likewise final-ineligible.
Exactly one already-selected exact-runtime pair/depth is evaluated on
final-heldout.

The ROCmFP4 FAST post-encoding audit now dispatches through the actual stored
destination type and its cross-decoder GPU-free regression passes. The arm is
still unpromoted until its exact intervened artifact passes the real-weight
gfx1151 differential, quality, memory, and performance gates below.

ROCMI4 W4A8 runtime evidence is control-correlated, not globally aggregated.
Each real-weight scope binds its exact target tensor at begin/end and requires
the ordered target route, IU4 launch (or MMVQ negative), and post-compute
completion. Other shared/dense/routed events from the real MoE graph may occur
inside that scope but cannot satisfy the target subsequence. Homogeneous ROCMI4
arms prove dense and routed controls, routed-FAST arms prove dense controls,
and broad FAST-matrix arms retain an explicit no-eligible-ROCMI4 not-applicable
mode. Clean finalist timing requests W4A8 only for a candidate with eligible
ROCMI4 MMQ weights and never enables dispatch telemetry or profiler counters.

Quality evidence is accepted only with the v2 offline judge contract and its
digest-pinned runtime-capture attestation. That attestation binds the stock,
candidate, judge, and agentic request/response indexes to the exact OCI image
reference, OCI config digest, Ember engine revision, hashed server executable,
hashed capture driver, model inventories, corpus, rubric, and judge artifact.
The verifier derives rating/severity consistency instead of trusting a
caller-supplied pass bit. Its release scope is explicitly text-only with
`multimodal_release_claim=false` and `vision_mmproj_differential_pass=false`;
the mmproj may be inventoried for memory accounting, but no multimodal quality
or release claim is permitted until a separate measured vision differential
passes.

Generate the live capture handoff with `scripts/qwen_quality_descriptor.py`.
The generator is GPU-free and stdlib-only: it reproduces the supplied
selection or MTP-depth-unlocked final plan, hashes both completed build records
and every ordered GGUF shard, verifies the checked rubric and agentic corpus,
and emits the exact `quality-capture-plan.v1` and `quality-phase-descriptor.v1`
objects consumed by `.github/workflows/qwen-quality-capture.yml`. The candidate
ID is explicit because it belongs to the construction receipt and is not stored
in the quantizer build record; final generation verifies it against the sealed
winner. The quality output root must not exist yet, and the generator does not
create it, start a runtime, quiesce production, publish, or delete anything.

The judge is supplied through a separately hashed, fail-closed inventory:

```json
{
  "schema": "ember.qwen3.8.quality-judge-inventory.v1",
  "artifact": {"path": "/abs/judge.gguf", "sha256": "<64-hex>", "bytes": 1}
}
```

For a selection-corpus capture, the invocation shape is:

```sh
python3 scripts/qwen_quality_descriptor.py \
  --phase sweep \
  --phase-plan /abs/qwen-bakeoff-plan.json \
  --phase-plan-sha256 '<64-hex>' \
  --stock-build-record /abs/stock/qwen-quant-build-record.json \
  --stock-build-record-sha256 '<64-hex>' \
  --candidate-build-record /abs/candidate/qwen-quant-build-record.json \
  --candidate-build-record-sha256 '<64-hex>' \
  --candidate-id '<construction-candidate-id>' \
  --judge-inventory /abs/judge-inventory.json \
  --judge-inventory-sha256 '<64-hex>' \
  --ember-revision '<40-hex>' \
  --model-runtime-image 'ghcr.io/otheru-ai/ember@sha256:<64-hex>' \
  --judge-runtime-image 'ghcr.io/otheru-ai/ember@sha256:<64-hex>' \
  --quality-output-root /abs/new-quality-capture \
  --capture-plan-output /abs/quality-capture-plan.json \
  --output /abs/quality-phase-descriptor.json
```

Use `--phase final` only with the canonical
`final_heldout_unlocked_after_mtp_depth_selection` plan. Both output JSON files
are create-only. Dispatch the workflow with the printed phase-descriptor path
and SHA-256; the workflow independently repeats every binding before taking the
GPU lock.

After the digest-matched stock ROCMI4 control and MTP companion exist, capture
directions with `scripts/qwen_capture_control.py`. The operator supplies every
image, build-record, shard, MTP, recipe, contract, and extraction-corpus digest
explicitly. The driver hashes large model artifacts with direct I/O, takes the
fixed GPU lock, stops and masks production, captures exactly 32 good plus 32
bad final-row activation records, and restores production before generating
the deterministic 4×4 intervention grid. It never opens the sweep or final
JSONL files and never quantizes or publishes anything. Use `--dry-run` first;
the printed plan is side-effect free. On the certification host the pinned
corpus directory is `/srv/ember/qwen3.8-otheru-corpus-a3c6a728`.

Each measured candidate is reduced to a digest-bound assessment, and the
externally attested ledgers are selected serially as `sweep`, `format`, then
`mtp-depth`. The complete format-arm sweep is only the screening measurement.
Before the format ledger can name a winner, its two highest-ranked gate-passing
arms require balanced confirmation in the plan-persisted counterbalanced order
A/B, B/A, A/B (`ABBAAB`). Three sealed workload recipes each run once per arm
as an adjacent pair. Both members calibrate the identical prompt construction,
then evict that calibration snapshot from the one-slot prefix cache before
timing; the retained full-prompt digests and calibrated word count must match
within each pair. Every slot starts a fresh server process and records one exact
2074-token prefill plus one 256-token decode, yielding exactly three samples per
arm. Selection requires the absolute decode-median lead to be strictly greater
than both the larger within-arm decode range and a predeclared 1.0 tok/s
practical-effect floor. One tok/s is about 2.5% at the 39.49 tok/s release
target; a smaller lead does not justify committing a different 128 GiB quant
artifact even when this six-run sample happens to have less observed spread.
At the format phase boundary the workflow invokes
`scripts/qwen_balanced_confirmation.sh`. The runner recovers each finalist only
through the externally attested assessment -> measured-result -> target
completion -> candidate-binding digest chain, then persists the top-two order
before acquiring the GPU. It holds the fixed GPU lock while production is
stopped and masked, starts and removes one server container for every slot, and
restores production on every exit path. It hashes the container ID, host PID,
process start tick, candidate/model/MTP identities, arm, and slot into the
unique process-instance identity. The full descriptor is retained alongside
its canonical SHA-256 and is checked against the runtime image and engine
binary. Each slot mounts and O_DIRECT-verifies the finalist's bound MTP,
enables its exact depth, and requires native speculation with
`0 < accept_rate < 1`. The timing container explicitly requests the compiled
IU4 path with `DFLASH_ROCMI4_W4A8_IU4=1`; each retained server log must contain
one recognized `w4a8_iu4_register_pack` or `w4a8_iu4_prepack` startup mode, and
all six slots must agree. Dispatch-evidence environment decisions are cached
once at process startup so disabled telemetry does not add per-dispatch
`getenv` overhead to clean timing.
The six confirmation slots are clean timing only; the screening gate's separate
profile/counter evidence is retained, and no profiler runs concurrently with
confirmation timing.
`unlock-final` accepts only the sealed MTP-depth ledger; it does not accept the
intermediate format ledger. The checked recipe uses schema v3, result evidence
v4, candidate assessments v2, and ledgers v3 so earlier records cannot be
mistaken for depth-bound evidence.

The stage commands follow this shape (each later command also supplies the
exact prior-ledger and attestation-bundle paths and SHA-256 values):

```sh
python3 scripts/qwen_bakeoff.py \
  --plan /root/qwen-work/qwen-bakeoff-plan.json \
  --stage mtp-depth \
  --results /root/qwen-work/qwen-mtp-depth-assessments.json \
  --prior-ledger /root/qwen-work/format-ledger.json \
  --prior-ledger-sha256 '<SHA256>' \
  --prior-attestation-bundle /root/qwen-work/format-ledger.bundle.json \
  --prior-attestation-bundle-sha256 '<SHA256>' \
  --output /root/qwen-work/mtp-depth-ledger.json
```

Every row must include at least three prefill and decode samples, audited
quality and differential results, the exact main/MTP/mmproj artifact byte
inventory, enabled companions, the exact 134,297,894,912-byte host MemTotal,
and measured peak RSS, GTT, and deduplicated accounted UMA bytes from the
runner sampler, including the exact live TTM `pages_limit`. The hard throughput
gates are the three-sample prefill peak at 412 tok/s and the three-sample decode
median at 39.49 tok/s.
Candidates must first pass differential correctness, audited quality, exact
measurement-shape/speculation, memory, prefill-peak, and decode-median gates.
Among only those passing candidates, each selection phase ranks decode median
descending, then prefill median descending, then audited quality score
descending, then stable candidate id ascending. The prefill peak remains a
hard threshold; it is not the prefill performance tie-break statistic.
For the final format choice, the first-stage order determines only the two
finalists. Both finalists must independently clear the unchanged 412.0 tok/s
prefill-peak and 39.49 tok/s decode-median gates during balanced confirmation.
The selector defines observed run noise conservatively as the larger of the
two within-arm decode ranges. The confirmation decode medians must differ by
strictly more than that value; equality or a smaller difference is
indistinguishable and rejects the selection instead of breaking the tie with
quality or identifier. It also computes the signed A-minus-B decode difference
for each adjacent pair, including the reversed middle pair; all three must have
the same nonzero sign as the overall median advantage. This rejects a winner
that changes with pair order, workload, or thermal sequence. The confirmation
object retains the premeasurement ordering/workload commitment, all six full
process descriptors and digests, raw samples, paired prompt hashes, derived
per-arm metrics, signed pair differences, separation, and observed-noise value.
Both the static artifact-plus-reserve-plus-companion sum and measured UMA peak
must fit real host MemTotal, which is stricter than the nominal 128 GiB
architectural budget. The decision remains non-publishing; release packaging
and hardware certification are separate gates.
