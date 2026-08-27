# Qwen3.8 serial candidate construction

Qwen intervention candidates are built from one immutable, content-addressed
BF16 cache. This is a disk and conversion-time constraint, not a shortcut in
the intervention: `ember-gguf-quantize` reads the cached stock BF16 tensors,
applies the exact direction manifest at its audited pre-encoding boundary, and
then encodes the selected mixed-format arm. No modified 360 GB BF16 duplicate
is materialized.

The main cache is deliberately “mostly BF16”: the patched bounded converter
stores the 204.8 GB PLE table as GGML F32. Converting that table to BF16 inside
llama.cpp materializes an additional roughly 102.4 GB anonymous result and was
measured to die at 128,524,508 KiB RSS on the 125 GiB control run. The builder
requires and records the one F32 PLE tensor; the final quantizer still proves
that the PLE override emits the selected quant type (ROCMI4 in every declared
arm). The separate vision mmproj remains BF16.

`scripts/qwen_candidate_builder.py` has four lifecycle boundaries:

1. `prepare-cache` performs the pinned, patched llama.cpp conversion with
   `--use-temp-file`, splits the main GGUF at 48G, and separately runs the
   pinned Qwen4Exp `--mmproj --outtype bf16` exporter. It hashes the main shard
   set plus mmproj into `bf16-<sha256>`, fsyncs it, and makes it read-only.
   The manifest also pins the patched converter, splitter, converter dependency
   lock, and exact builder-container digest.
2. `make-companion-inventory` binds that shared BF16 mmproj to one exact MTP
   export. MTP matrix encoding must match the selected main-model arm; use
   `scripts/qwen_mtp_export.py` once for each actually tested MTP matrix
   contract. An undeclared or unsupported arm is rejected rather than mapped
   to a convenient format.
3. `build-candidate` revalidates the full cached shard hashes, mmproj, pinned
   converter/patch/splitter identities, intervention, MTP inventory, quantizer
   build information, and the live TTM cap. It then creates exactly one
   candidate under the exclusive workset lock. The command records the builder
   image digest and builder revision separately from runtime-engine identity;
   graph/kernel-only runtime revisions can reuse identical model bytes when
   their tensor-format compatibility contract matches.
4. `record-assessment` fsyncs an exact build-record, assessment, and shard
   inventory outside the candidate directory. Only a bundle marked unselected
   authorizes `delete-loser --execute`. Deletion removes only the inventoried
   quant shards and retains the build/intervention evidence plus an external
   deletion tombstone. It is not recoverable.

Both conversion and candidate encoding require a cgroup-v2 boundary with
`memory.max=134217728000` and `memory.swap.max=0`. The builder records
`memory.peak` and child maximum RSS and fails closed above the limit. Run each
operation in a fresh `docker --memory 125g --memory-swap 125g` container so
`memory.peak` describes that operation. Cache conversion additionally requires
at least 1152 GiB free disk. The workset lock prevents a second
BF16 conversion, intervention, quantization, or deletion process from running
concurrently.

On the gfx1151 certification host, preparation and candidate construction must
be wrapped by `/usr/local/sbin/ember-gpu-lock` and
`/usr/local/sbin/ember-cert-production`, including unconditional unmask,
service-health verification, and lock release on every exit path. The builder
does not stop production itself and must not be invoked directly around the
fixed-purpose host boundary.

The builder never uploads or publishes. The immutable cache is construction
input only; a release contains the selected quant shards, matching MTP,
mmproj, build records, intervention manifest, and later hardware evidence.

The dedicated gfx1151 bakeoff workflow consumes one
`ember.qwen3.8.sequential-bakeoff-candidate.v3` manifest per dispatch. The
manifest keeps the artifact-builder revision/image separate from the runtime
engine revision/image/binary and binds both to one tensor-format compatibility
contract. Sweep dispatches can open only the selection corpus. A final
candidate must carry no final corpus digest: `qwen_bakeoff.py --stage
unlock-final` verifies the externally-attested format ledger before the
workflow opens final-heldout.

After production is active and healthy again on port 8000, the workflow runs
`--stage assess` while the candidate shards still exist, GitHub-attests and
locally verifies the compact assessment, and appends only its attestation
descriptor to an attested accumulator. Phase ledgers are likewise attested and
verified. A proven hard-gate failure or a nonwinner in a completed phase ledger
may then pass through `record-assessment` and `delete-loser`; an as-yet passing
candidate is retained provisionally because it is not truthful to call it a
loser before the selector sees the complete phase. Operators must provide
enough artifact spool for that provisional survivor, or explicitly reconstruct
it from the immutable cache and pinned build record before final confirmation.
