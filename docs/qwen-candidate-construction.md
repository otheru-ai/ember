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

`scripts/qwen_candidate_builder.py` has eight lifecycle boundaries:

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
3. `retire-captured-stock` can free the direct-conversion stock shards after
   activation capture.  It first fsyncs an exact authorization outside the
   stock directory, verifies every shard against both the stock build record
   and capture manifest, deletes only those inventoried shard inodes, retains
   the build record, and fsyncs a completion tombstone.  The stock control is
   then rebuilt from the immutable cache and must reproduce the captured shard
   sizes and hashes exactly.
4. `build-candidate` revalidates the full cached shard hashes, mmproj, pinned
   converter/patch/splitter identities, intervention, MTP inventory, quantizer
   build information, and the live TTM cap. It then creates exactly one
   candidate under the exclusive workset lock. The command records the builder
   image digest and builder revision separately from runtime-engine identity;
   graph/kernel-only runtime revisions can reuse identical model bytes when
   their tensor-format compatibility contract matches.
5. `record-assessment` fsyncs an exact build-record, assessment, and shard
   inventory outside the candidate directory. Only a bundle marked unselected
   authorizes `delete-loser --execute`. Deletion removes only the inventoried
   quant shards and retains the build/intervention evidence plus an external
   deletion tombstone. This irreversible legacy boundary is not used by the
   gfx1151 bakeoff.
6. `authorize-rolling-retention` verifies the GitHub attestations over the
   compact accumulator and every referenced assessment, rederives the selector
   metrics from the exact selection plan, and emits only the newly displaced
   candidate. Sweep retains the stock control plus its current top one; format
   retains the two final-eligible arms needed by balanced confirmation.
7. `authorize-sealed-retention` runs only after the completed format ledger is
   GitHub-attested and verified. It reproduces the balanced selector, retains
   its one winner, and authorizes retirement of the other live finalist.
8. `retire-reconstructable` fsyncs an authorization outside the candidate,
   proves the content-addressed BF16 cache and companions still exist, and
   atomically quarantines and re-verifies each exact inventoried shard before
   unlinking it. Its deterministic quarantine state is resumable after an
   interrupted rename or unlink. `restore-reconstructable` accepts only a fresh, distinct,
   same-filesystem rebuild with the same immutable construction contract and
   every original shard size/hash; no-clobber renames are resumable after an
   interrupted restore, and the final receipt remains outside both directories.

Both conversion and candidate encoding require a cgroup-v2 boundary with
`memory.max=134217728000` and `memory.swap.max=0`. The builder records
`memory.peak` and child maximum RSS and fails closed above the limit. Run each
operation in a fresh `docker --memory 125g --memory-swap 125g` container so
`memory.peak` describes that operation. Cache conversion additionally requires
at least 1152 GiB free disk. On the certification host the direct stock control
does not fit beside that preflight margin, so it must be activation-captured
and retired through the durable lifecycle above before cache preparation; an
ad-hoc `rm` is not equivalent evidence. The workset lock prevents a second
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

## Durable captured-stock retirement

[`qwen-gfx1151-retire-stock.yml`](../.github/workflows/qwen-gfx1151-retire-stock.yml)
is the only supported destructive boundary for the activation-captured stock
control. A dispatch supplies the full Ember commit, exact development-image
reference and OCI digest, the independent full revision that produced the stock
artifact, the fixed gfx1151 workset, exact stock build-record and
capture-manifest digests, a new authorization path, and the literal
`RETIRE_CAPTURED_STOCK_SHARDS` acknowledgement. `commit_sha` identifies only
the checked-out retirement implementation and its pinned builder image;
`stock_artifact_revision` identifies the already-captured stock bytes and may
legitimately differ. For the current control it is
`ebf39327900c91354157910c4401ca96da3b688b`, which derives
`stock-rocmi4-ebf39327900c`.

The workflow refuses noncanonical paths: the stock control must be the
artifact-revision-named direct-conversion directory, the capture must record
that same artifact revision and be the persistent `capture-manifest.json` under
the Qwen evidence root, and all retirement records live outside the stock
directory under the fixed workset evidence root. The operator chooses a fresh
safe `stock-*-authorization.json` basename so a failed or completed attempt can
never be overwritten by a later dispatch.

Before production is quiesced, the workflow fsyncs an immutable dispatch
authorization containing the exact shard deletion scope and hashes of every
stock-directory file that must remain. Under the shared gfx1151 lock it stops
and masks production, then runs `qwen_candidate_builder.py
retire-captured-stock` in the digest-pinned, networkless, read-only container.
Only the stock directory, workset, exact capture file, and read-only checkout
are mounted. The builder independently verifies the build record, capture, and
every shard, fsyncs its authorization, unlinks only the inventoried shard
inodes, retains the build record, and fsyncs its completion record.

Post-retirement verification requires both builder records to match the
dispatch deletion scope byte-for-byte, proves every authorized shard is absent,
rehashes every retained stock file plus the capture manifest, and fsyncs a
workflow completion envelope. Production is unmasked and started, the lock is
released, and port 8000 health is proved on the unconditional cleanup path.
The workflow has read-only repository/package permissions and performs no
publication or artifact upload; all evidence remains runner-local.

Retirement is recoverable only by reconstruction. The deleted inode contents
cannot be undeleted in place. Recreate the byte-identical stock control from the
pinned source snapshot through the content-addressed BF16 cache and audited
quantizer, retaining the authorization/completion chain as the reason the
direct-conversion shards are absent.

The dedicated gfx1151 bakeoff workflow consumes one
`ember.qwen3.8.sequential-bakeoff-candidate.v3` manifest per dispatch. The
manifest keeps the artifact-builder revision/image separate from the runtime
engine revision/image/binary and binds both to one tensor-format compatibility
contract. Sweep dispatches can open only the selection corpus. A final
candidate must carry no final corpus digest: `qwen_bakeoff.py --stage
unlock-final` verifies the externally-attested MTP-depth ledger before the
workflow opens final-heldout.

The bakeoff dispatch likewise remains below GitHub's ten-input limit. Its
`phase_request` path/SHA names an exact
`ember.qwen3.8.sequential-bakeoff-phase-request.v1` object with exactly
`{schema, phase, results_accumulator, prior_ledger, publishes, deletes}`.
Each non-null evidence value has an exact `{subject, bundle}` pair; the subject
binds absolute path, SHA-256, and accumulator-v2 or ledger-v3 schema, while the
bundle binds its absolute path and SHA-256. The workflow still verifies both
files through GitHub attestation against `OtherU-AI/ember` and this bakeoff
workflow. Sweep forbids a prior ledger; every later phase requires the
immediately preceding phase ledger.

After production is active and healthy again on port 8000, the workflow runs
`--stage assess` while the candidate shards still exist, GitHub-attests and
locally verifies the compact assessment, and appends only its attestation
descriptor to an attested accumulator. The builder independently verifies the
accumulator and assessment attestations before deriving a prefix transition.
This is not an early winner declaration: a candidate leaves the live set only
when it cannot re-enter the required top set after later rows are appended.
Sweep therefore holds the stock control plus one intervention candidate, and
format holds exactly two final-eligible candidates through balanced
confirmation. At the format boundary, only the verified completed ledger may
collapse those two artifacts to its selected winner. Every eviction retains a
reconstructable authorization/completion chain; the bakeoff never calls the
irreversible `delete-loser` path.

All per-dispatch measurements, assessments, attestations, accumulators,
authorities, ledgers, and retirement records are run/attempt scoped. A retry
therefore cannot overwrite evidence already named by an authorization. If the
earlier attempt reached retirement, restore the candidate through
`restore-reconstructable` before asking the workflow to measure that row again;
the workflow does not pretend an absent quant artifact can be remeasured.

## Checked-in gfx1151 construction workflow

[`qwen-gfx1151-construct.yml`](../.github/workflows/qwen-gfx1151-construct.yml)
is the supported orchestration boundary after stock activation capture. GitHub
limits `workflow_dispatch` to ten inputs, so every dispatch supplies only the
commit, mode, and an exact path/SHA pair for an
`ember.qwen3.8.candidate-construction-request.v1` manifest. The request has
exactly `{schema, mode, parameters, publishes, deletes}`; its parameter keys
are exact and mode-specific.

[`qwen-gfx1151-request-bridge.yml`](../.github/workflows/qwen-gfx1151-request-bridge.yml)
is the non-GPU bridge for creating that first runner-local request. It accepts
the complete JSON bytes as strict base64 plus their SHA-256, validates the
exact mode-specific schema and rehashes every referenced descriptor, and uses
`O_EXCL` plus file and directory fsync beneath the fixed
`qwen-workset/evidence/operation-requests` directory. Its summary reports the
exact persistent path and digest consumed by the construction workflow. It
does not quiesce production, touch model bytes, publish, delete, or overwrite.

Until these specialized workflows land on the default branch, dispatch them
through the default-branch `gfx1151-certify.yml` entrypoint at the exact target
ref with `release_version=qwen-dispatch`. Its two additional inputs carry one
strict-base64 `ember.qwen3.8.branch-dispatch-envelope.v1` object and the
SHA-256 of its decoded bytes. The envelope binds the same full Ember revision,
one of `request`, `construct`, or `retire`, exact operation inputs, and explicit
non-publication/deletion lifecycle. Three static local reusable-workflow calls
then select the operation. GitHub resolves a `./.github/workflows/...` reusable
workflow from the caller's same commit, so branch logic cannot drift from the
`commit_sha` supplied to the dispatcher. Retirement keeps its literal
`RETIRE_CAPTURED_STOCK_SHARDS` acknowledgement inside the digest-bound
envelope. The default dispatcher has four inputs, and every called workflow
has at most ten.
The decoded outer envelope is limited to 32 KiB and a nested construction
request to 16 KiB, leaving headroom beneath GitHub's 65,535-character total
manual-input payload limit after base64 expansion.

The four serial dispatch modes are:

1. `prepare-cache`, which creates the single content-addressed cache;
2. `prepare-companions`, which creates both homogeneous `Q4_0_ROCMI4` and
   `Q4_0_ROCMFP4_FAST` MTP exports, binds each to the shared BF16 mmproj in a
   canonical `{schema, source, companions}` inventory, and writes the verified
   selection plan. Its post-operation summary reports an exact path/SHA-256
   pair for the durable companion-construction descriptor and for both
   inventories plus the selection plan. The descriptor is fsynced and binds
   all three child path/digest pairs with `publishes:false` and `deletes:false`;
3. `build-candidate`, which builds exactly one stock or intervention candidate.
   The stock dispatch consumes the exact activation-capture manifest and must
   reproduce its shard bytes from the cache before it receives a builder
   attestation;
4. `normalize-candidate`, which consumes one exact
   `ember.qwen3.8.candidate-normalization-request.v1` descriptor and emits a
   new v3 candidate manifest without taking the GPU or quiescing production.

For example, normalization is dispatched with this outer request (all paths
are absolute runner paths and every digest is lowercase SHA-256):

```json
{
  "schema": "ember.qwen3.8.candidate-construction-request.v1",
  "mode": "normalize-candidate",
  "parameters": {
    "normalization_request": {"path": "/abs/normalize.json", "sha256": "<64-hex>"}
  },
  "publishes": false,
  "deletes": false
}
```

The referenced normalization request contains exact descriptor-or-null fields
for `construction`, `quality_contract`, `prior_accumulator`, `prior_ledger`,
and `final_plan`; explicit `stage`, `row_id`, `mtp_matrix_quant_contract`,
`mtp_depth`, and `runtime_mode`; an exact runtime object with revision,
release/dev OCI refs and digests, and tensor-format contract SHA; a new
absolute `output`; and literal `publishes:false`/`deletes:false`. Stock/sweep
forbid prior ledgers, format requires the sweep ledger, MTP-depth requires the
format ledger, and final requires both the MTP-depth ledger and the separately
unlocked final plan. A prior accumulator is optional only when the row is first
in its canonical phase order.

Every sequential bakeoff summary similarly reports the exact subject and
attestation-bundle path/SHA-256 pairs for the new compact accumulator. At a
completed phase boundary it also reports both pairs for the new phase ledger,
so the next digest-bound `phase_request` can be assembled without inferring a
digest from a filename or from console output.

The workflow resolves the matching development and release tags to exact OCI
repository digests and runs subsequent containers by `repository@sha256:...`.
It pins the Qwen snapshot, llama.cpp, ROCmFPX, PLE patch, Qwen4Exp converter,
and frozen converter environment. Persistent paths are mounted at the same
absolute host/container locations, and every construction container runs as
the runner UID/GID in a fresh 125 GiB/no-swap cgroup. It refuses symlinked or
pre-existing outputs. This workflow never calls either candidate retirement or
loser deletion and never publishes an artifact.

Cache preparation still requires the measured-path floor of 1152 GiB free
(1536 GiB preferred). Candidate shards consume additional persistent space,
and the workflow does not clear space automatically. The fixed host lock and
production wrapper cover only the exclusive operation; cleanup always unmasks
and starts production, releases the lock, and proves the port 8000 health check.

## Construction-to-measurement descriptor

Each candidate build writes a durable
`ember.qwen3.8.candidate-construction.v1` descriptor. It binds the exact cache,
both companion inventories, selection plan, activation capture, intervention,
build record, builder attestation, OCI identities, quant arm, and shard
inventory. It deliberately records `v3_candidate_manifest.ready=false`.

The descriptor is not accepted directly by the bakeoff workflow.
`scripts/qwen_candidate_manifest.py from-request` adds and validates the
phase-specific audited quality contract, current accumulator/ledger state,
selection-corpus binding, complete legacy gate fields, and runtime identity
before emitting `ember.qwen3.8.sequential-bakeoff-candidate.v3`. Construction
remains `stock`/`sweep`/`format`: MTP-depth rows reuse the exact format-selected
main and companion bytes while changing only depth, and final confirmation
reuses the exact MTP-depth-selected identity. The independently selected MTP
matrix contract therefore comes from the format arm, not from the main-model
quant arm.

ROCMI4 W4A4 is a final-ineligible auxiliary runtime control over exact ROCMI4
artifact bytes. It is intentionally outside the serial exact-runtime winner
ledger and cannot be normalized as a selectable format/MTP-depth/final row.
Any future auxiliary runner must bind separately built W4A4 release/dev images
and must never inherit the ordinary exact-dequant runtime identity.
