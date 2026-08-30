# Qwen3.8 Flash Next Heretic ROCmI4 release procedure

This procedure covers an offline candidate package for
`otheru/Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-GGUF`. It does not claim that
Ember can load or serve the model before the Qwen runtime work and target-host
certification land. The checked-in release profile pins the audited inputs:

- base model `Qwen/Qwen3.8-Flash-Next` at
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`;
- ROCmFPX at `c49ebdbd5c9f01ec242369f9e7f7967855f80cba`;
- OtherU's intervention pipeline branch `ember-contract-and-drafter-fix` at
  `a3c6a728510f91394e991504951ac316cd3a89af` and upstream Heretic at
  `bedb94ef117a271532ac2058447fbc165d5051bd`;
- storage type `Q4_0_ROCMI4` (GGML type 108, 4.25 bits per weight); and
- the Qwen Community License 1.0 file at the pinned source revision, including
  its SHA-256 digest.

`ROCmI4` is the quantized storage format. `IU4` is the gfx1151 dot-product
instruction used by a compute path; it is not a second file format. The normal
exact-dequant path and `GGML_HIP_ROCMI4_W4A4=ON` are distinct release modes.
W4A4 is lossy, remains off by default, and needs an independent quality report
and benchmark. Published ROCmFPX results from other models or workloads are
useful hypotheses only, never Qwen3.8 Flash Next or Ember performance claims.

“Heretic” is a weight-provenance claim in this release. A prompt-only system
message, chat-template change, runtime steering flag, or ordinary stock-model
quant is not eligible for that name. The quantizer applies a signed, completed
directional-ablation manifest while each selected BF16/F32 tensor is encoded to
ROCmI4 and reports the exact manifest and target-map digests. The packager
rejects any build that lacks that application evidence.

## Required intervention manifest

`--intervention-manifest` is mandatory for both planning and execution. The
manifest is schema 1, `status: complete`, `kind: directional_ablation`,
`weight_intervention: true`, `prompt_only: false`, and
`application_stage: pre_quantization_encoding`. It binds the pinned base model
and snapshot inventory to two tool inputs:

- `otheru-quant-pipeline` branch `ember-contract-and-drafter-fix` at
  `a3c6a728510f91394e991504951ac316cd3a89af`;
- upstream `p-e-w/heretic` at
  `bedb94ef117a271532ac2058447fbc165d5051bd`.

The manifest contains the direction-extraction corpus count and SHA-256 with a
zero held-out-evaluation overlap assertion, inline finite F32 direction values
and their packed little-endian SHA-256, and a lexicographically ordered exact
tensor map. Regex, suffix,
or layer-range targets are not accepted. Qwen's residual writers are
architecture-specific: QSA layers 3, 7, 11, …, 47 use
`blk.N.attn_output.weight`; all other layers 0–47 use
`blk.N.ssm_out.weight`. Vision, PLE, router, expert, and MTP tensors are outside
this intervention contract. Each target records its direction id, non-zero
scale, `row_norm_preserve`, and GGUF-order shape `[ne0 columns, ne1 rows]`; the
direction length must equal `ne1`.

The quantizer hashes the exact manifest bytes it parses. Dry-size preflight
must report the manifest hash, ordered targets, target count and tensor-map
hash with `intervention_validated: true` and `intervention_applied: false`.
The real conversion must return the same evidence with
`intervention_applied: true`. Every output shard independently carries those
hashes and the target count in `ember.intervention.*` GGUF metadata. This is
what prevents a stock quant, or a prompt-only variant, from being packaged
under the Heretic repository name.

Application also records finite per-target projection and distortion metrics:
source/stored refusal-projection L2, their ratio, signed projection
coefficient, relative Frobenius delta, and row-norm relative RMSE/maximum.
These are measured after ROCmI4 storage, because row-norm preservation can
reintroduce part of a removed direction and quantization can add leakage. The
packager requires the metrics in exact target order and retains them in the
artifact manifest; threshold selection remains a quality-gate decision based
on the held-out corpus, not a hard-coded claim in this provenance check.

## External comparison recipes and provisional prior art

The pinned upstream Qwen model card provides SGLang and vLLM launch recipes
with tensor parallelism 4, a 262,144-token context, the `qwen3` reasoning
parser, the `qwen3_coder` tool parser, MTP speculative decoding, and three
speculative tokens. Its four-GPU AMD Instinct MI355X vLLM recipe sets
`VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_MOE=0`; the PLE CPU-offload
variant additionally sets `VLLM_PLE_CPU_OFFLOAD=1` and requires DEP.

TokenSpeed documents optional Hugging Face overrides
`ple_embed_dtype=float8_e4m3fn` and
`index_share_for_mtp_iteration=true`. These are approximations and remain
opt-in, never release defaults. All of these settings are preserved in the
release profile and generated manifest as `external_comparison_baseline`.
They describe other frameworks and hardware, not measurements or validated
settings for Ember on Strix Halo.

The official Qwen blog describes a 125B-parameter main model plus a 51B
n-gram embedding, with about 6B parameters active per token. Its native context
is 262,144 tokens; one million tokens is an optional YaRN extension, not a
native-context claim. PLE is designed for asynchronous host-memory prefetch,
and residual weights may optionally remain FP8. Those facts guide memory and
offload experiments but do not certify Ember's placement or throughput.
The checked-in profile records a passing scalar/math oracle for the YaRN
one-million-token equations, but no real-weight target differential has run:
`yarn_1m_runtime_certified` and `yarn_1m_fit_claim` therefore remain false.
One million tokens is correctness investigation only, not a Strix Halo fit
claim or a release gate satisfied by the current artifact.

The blog's managed `qwen3.8-flash`/Codex example is a distinct production
service with text and image input, original image detail, parallel tools,
reasoning levels `xhigh`/`medium`/`low`, and a one-million-token cloud context.
Do not attribute those managed-service capabilities to this open-weight GGUF.
Similarly, ClawEval-MM, RecreationBench, AndroidWorld, OSWorld 2.0,
Vision2Web, and ERQA are external vision-quality comparison context. Ember must
run and publish its own pinned cases before making any corresponding claim.

Unsloth's [Qwen3.8 Next comparison](https://unsloth.ai/docs/models/qwen3.8-next.md)
is also recorded as external quality/size context, not Ember evidence:

| External quant | Size (GB) | KLD | Same-top (%) |
| --- | ---: | ---: | ---: |
| UD-Q4_K_XL | 111.3 | 0.044715 | 93.481 |
| UD-IQ4_XS | 93.7 | 0.079162 | 91.089 |
| Q3 | 90.0 | 0.099694 | 90.387 |
| IQ3 | 82.0 | 0.156505 | 87.570 |
| Q2 | 78.9 | 0.213343 | 85.163 |
| IQ1_M | 74.5 | 0.302159 | 82.396 |
| IQ1_S | 72.5 | 0.375140 | 80.239 |

That source keeps random-access n-gram/PLE tensors at four bits or higher for
quality. These results use different formats and tooling; they are comparison
points for the hard 128 GiB UMA target, not proof that an Ember ROCmI4 build
fits, meets the same quality, or achieves any throughput.

Two independently published ROCmFP4 Strix Halo artifacts are pinned as
additional size/layout baselines. This audit read only Hugging Face API
metadata and the model cards; it did not download or inspect GGUF weight bytes.
Artifact sizes and hashes below are API manifest facts, while tensor policies
and measurements are publisher claims:

| External artifact | Revision | Layout | Exact bytes | Card-reported policy | Card generation |
| --- | --- | --- | ---: | --- | ---: |
| [STRIX_LEAN](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-GGUF/tree/dec9c5c1053ef814cfaa39b342efd4cdd721ef0b) | `dec9c5c1053...` | 3 shards: 44,944,111,488 + 44,688,348,576 + 16,121,070,688 | 105,753,530,752 | ROCmFP4 family attention/MoE; PLE Q5_1; token embedding Q5_K; head Q6_K; 4.78 bpw | 24.1 tok/s |
| [STRIX](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX-GGUF/tree/976378158e6005da4152e98eb672f71f8bd5265c) | `976378158e60...` | 1 file | 121,838,036,032 | attention K/V base ROCmFP4; remaining/MoE ROCmFP4 family; PLE Q8_0; embedding/head Q6_K; 5.51 bpw | 14.5 tok/s |

The LEAN shard SHA-256 values are `50830ad9...adf0`,
`30abc400...a58e`, and `97147d58...9727`; STRIX is
`c6770d74...49fa`. Both measurements are described as single-stream greedy on
a Ryzen AI MAX+ 395/Radeon 8060S, gfx1151, ROCm 7.2.4, with all 49 layers
offloaded, `-ngl 999`, `--fit off`, `--no-warmup`, `--ctx-size 2048`, and 16
threads. Prompt processing was not measured, and the cards omit a benchmark
prompt, generated-token count, warmup protocol, runtime commit, and quality
corpus, so these are incomplete external observations rather than Ember
performance or quality evidence.

Neither artifact passes this release's native-262K gate. LEAN plus the pinned
32 GiB reserve totals 140,113,269,120 bytes, exceeding 128 GiB by
2,674,315,648 bytes; STRIX totals 156,197,774,400 bytes, exceeding it by
18,758,820,928 bytes. Ember's vendored engine contains ROCmFP4 storage/runtime
types, but this release pipeline deliberately accepts only ROCmI4 type 108 and
requires its own build record, exact-dequant certification, and target tests.
ROCmFP4 results do not validate the signed-IU4 ROCmI4 artifact, the optional
gfx1151 W4A4 path, or frontier WMMA behavior.

Pinned provisional runtime references are also recorded, but are not release
evidence: `radicalgeek/ROCmFPX` branch
`halospeckv/accepted-prefix-replay` at
`60ff854bdc25e27ee211ac0c4df896e9379edd3f` is gfx1151 Qwen3.8 27B MTP
strict-replay prior art, not Flash Next validation. The open, unmerged llama.cpp
Qwen4Exp base PR #27742 is pinned at
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`; its open rotated-KV/QSA fix PR
#27774 is pinned at `abdc7a0bf815d3b83e26dd523c6960e4dd597e82` and applies a
13-line `qwen4exp.cpp` delta: Hadamard rotation of Q/K/V before sparse attention
and inverse V-path rotation after `build_attn_mha`. Neither unmerged PR is an
upstream release or a substitute for Ember certification.

## Reproducible conversion and quantization

The orchestrator is stdlib-only and validates the complete 144-file Hugging
Face snapshot against the checked-in inventory: every path, size, LFS SHA-256
or Git blob, the pinned license, config architecture, and exact revision. It
also validates clean source checkouts, the PR #27742/#27774 parent relationship,
ROCmFPX provenance, host resources, and the Ember streaming quantizer's
machine-readable build identity. The default writes
`/scratch/qwen3.8-rocmi4.plan.json` and performs no conversion. It deliberately
does not create the final work directory, so the same command can subsequently
be rerun with `--execute`:

```bash
python3 scripts/qwen_quantize.py \
  --snapshot-dir /models/Qwen3.8-Flash-Next/snapshots/f5d08274bafd880402bd16f5e3e6c514136ec06c \
  --snapshot-revision f5d08274bafd880402bd16f5e3e6c514136ec06c \
  --intervention-manifest /models/qwen3.8-heretic/intervention-manifest.json \
  --llama-cpp-dir /src/llama.cpp-qwen4exp \
  --rocmfpx-dir /src/ROCmFPX \
  --ember-dir "$PWD" \
  --ember-revision "$(git rev-parse HEAD)" \
  --quantizer build/ember-gguf-quantize \
  --work-dir /scratch/qwen3.8-rocmi4 \
  --threads "$(nproc)"
```

Add `--execute` to run the two recorded commands. Conversion uses the pinned
llama.cpp Qwen4Exp head `abdc7a0b...`, whose direct parent is the audited
PR #27742 head `035e2273...`. Quantization uses Ember's architecture-agnostic
streaming GGUF quantizer plus the pinned ROCmI4 implementation provenance at
ROCmFPX `c49ebdb...`. This composite is necessary: upstream llama.cpp has
Qwen4Exp but no ROCmI4, while ROCmFPX at that revision has ROCmI4 but no
Qwen4Exp model loader. Neither upstream alone is a working pipeline.

The quantizer receives `--tensor-type
'^per_layer_token_embd\.weight$=Q4_0_ROCMI4'` before every positional argument,
followed by `--intervention-manifest` and the profile-pinned
`--device-budget-bytes 137438953472` and
`--runtime-reserve-bytes 34359738368`, then the input, output,
`Q4_0_ROCMI4`, and thread count. Before the real quantization command, the
orchestrator invokes the same command with `--dry-size-json`, consumes its
exact JSON, and refuses to continue unless `artifact_bytes + 34359738368 <=
137438953472`. For split output the JSON also records `shard_count` and the
ordered `shard_bytes` list; their exact sum is `artifact_bytes`, and the
orchestrator compares every final shard size with that list before completing
the build record. The 32 GiB non-artifact reserve is provisional until target
peak-RSS and GTT measurements replace it; the JSON and resulting headroom are
retained in `qwen-quant-build-record.json`. ROCmFPX's upstream
parser stops option processing at its first positional, so moving the override
later can silently leave the enormous PLE row-gather table in the wrong type.
The positional quant type selects ROCmI4 for eligible model weights; the build
record verifies that the PLE table is GGML type 108, records the full output
type histogram, preserves the source I64 PLE hash constants exactly as
`ARRAY<UINT64>` GGUF metadata, compares pre/post tensor inventories, and hashes
every output.
W4A4 is not a conversion mode here; it remains a separately certified runtime
opt-in.

For clarity, the generated quantizer calls have this shape (the orchestrator
substitutes the pinned paths and exact thread count):

```bash
build/ember-gguf-quantize \
  --tensor-type '^per_layer_token_embd\.weight$=Q4_0_ROCMI4' \
  --intervention-manifest /models/qwen3.8-heretic/intervention-manifest.json \
  --keep-split \
  --device-budget-bytes 137438953472 \
  --runtime-reserve-bytes 34359738368 \
  --dry-size-json \
  /scratch/qwen3.8-rocmi4/Qwen3.8-Flash-Next-BF16-00001-of-000NN.gguf \
  /scratch/qwen3.8-rocmi4/Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo.gguf \
  Q4_0_ROCMI4 "$(nproc)"
```

After that preflight passes, the executed call omits `--dry-size-json` and runs
entirely inside a unique private sibling of the requested work directory. This
keeps verification and publication on one filesystem without exposing
unverified files at their final paths. Execute mode requires `--work-dir` to be
absent; an existing directory, file, or dangling symlink is a hard conflict.

The default conversion-runner resource floor is 1152 GiB free disk and 256 GiB
physical RAM. The conservative RAM floor covers the pinned converter's roughly
204 GB intermediate representation; 128 GiB is not a supported conversion
runner. This host-RAM floor is distinct from the 128 GiB UMA artifact
plus runtime-reserve gate above. The release default is a 48G split BF16
intermediate (`--split-max-size 48G`). Given its first
`NAME-00001-of-000NN.gguf` shard, `--keep-split` requires every ordered sibling,
validates `split.no`, `split.count`, `split.tensors.count`, and a unique global
tensor inventory, then preserves the llama.cpp filenames and split metadata in
the quantized output. Pinned llama.cpp stores complete model/provenance metadata
only in intermediate shard 1; every intermediate continuation contains exactly
the three split locator fields. Native ROCmFPX `--keep-split` at the pinned
revision follows that first-shard-only convention, but the final writer is
Ember's streaming quantizer. It copies each input shard, stamps
`general.quantization_version=2` and `general.file_type=118` onto every output,
and stamps either the two exact stock-control labels or all five exact
directional-ablation evidence fields onto every output. Final continuation
shards must contain precisely those mode-specific fields plus the split
triplet; the verifier rejects missing, extra, mistyped, or mismatched values.
The manual PLE regex is matched across the global set, so it need only occur in
its owning shard.

Planning all shards and the aggregate 128 GiB fit decision completes before
publication. The orchestrator rechecks authoritative shard-1 metadata and PLE
constants, every shard's split locators, the global tensor-name inventory, type
histogram, ordered sizes, and hashes in private. It removes the BF16
intermediate, writes and syncs the completed build record beside the verified
shards, syncs the private directory, then publishes that whole directory in
one Linux
`renameat2(RENAME_NOREPLACE)` operation. Shards and their complete provenance
record and the exact applied `qwen-intervention-manifest.json` therefore become
visible together. No check-then-unlink rollback is
used; POSIX cannot make such a conditional deletion race-free. A failed
quantizer, semantic check, record write, or directory commit exposes no final
artifact directory and cleans only the private transaction directory.

Rerunning `--execute` against a completed work directory leaves the committed
directory byte-for-byte unchanged. The standalone C++ quantizer has a weaker
CLI-shaped boundary because its output is a set of final filenames: it creates
a persistent O_EXCL transaction marker, retains its marker file descriptor,
and changes the marker contents to `COMPLETE` only after all no-clobber links
succeed. On a late conflict it exits nonzero and intentionally retains every
owned partial plus the incomplete marker for explicit recovery; it never tries
to delete a possibly replaced final name. Missing siblings, inconsistent
metadata, duplicate tensors, changed ordered sizes, and incomplete sequences
still fail closed before publication.

The llama.cpp converter streams PLE shards through a temporary mapping but its
current intermediate representation can consume roughly 204 GB. Do not add a
direct Python ROCmI4 outtype by name alone: ROCmFPX's Python class currently
implements dequantization but not quantization. A future direct converter must
first implement the C-reference UE4M3/nibble rounding exactly and quantize each
160-value PLE row to 85 bytes before mapping it.

## Offline candidate package

### Separate vision provider and BF16 mmproj

The pinned Qwen converter excludes `model.visual.*` from the normal text
artifact. Produce the vision tower in a separate converter pass with
`--mmproj --outtype bf16`; the required release filename is
`Qwen3.8-Flash-Next-BF16-mmproj.gguf`. The release image builds the dynamic
provider against llama.cpp revision
`abdc7a0bf815d3b83e26dd523c6960e4dd597e82`. To reproduce that provider outside
the image (the CPU mode is only a no-GPU linkage check):

```bash
scripts/build_qwen_vision_provider.sh \
  --build-dir build-qwen-vision-provider \
  --install-dir build-qwen-vision-provider/install \
  --backend hip
```

Set `DFLASH_QWEN_VISION_MMPROJ` and `DFLASH_QWEN_VISION_TEXT_MODEL` to the
packaged BF16 projector and matching vocab-only GGUF paths for image requests.
The image supplies `DFLASH_QWEN_VISION_PROVIDER`. Both the shared libraries and
the vocab-only text-model view are opened lazily on the first image encode, so
text-only generation has no provider residency or dispatch cost and never
loads duplicate text tensors. Data-URL PNG/JPEG/WebP/GIF still images are
accepted; remote HTTP image URLs and video remain unsupported and fail closed.
GPU-free ABI/dependency coverage alone is not certification. Run the protected
`qwen-gfx1151-vision.yml` workflow against the exact selected model, MTP, BF16
mmproj, vocab-only GGUF, release image, and matching dev image. It compares Ember's cold and
warm embedding rows for the pinned two-image corpus with a standalone oracle
built directly against llama.cpp
`abdc7a0bf815d3b83e26dd523c6960e4dd597e82` (float32 `atol=1e-5`,
`rtol=1e-5`), while separately sampling RSS, GTT, and total UMA. The gate owns
the GPU exclusively and restores production before emitting
`vision-certified.json`.

Run the generator only after producing the GGUF in a pinned build container and
copying `LICENSE` from the pinned source snapshot. Include the completed build
record so package hashes are tied to the verified conversion:

```bash
python3 scripts/qwen_release_package.py \
  --profile share/release_profiles/qwen3.8-flash-next-rocmi4-strix-halo.json \
  --artifact /scratch/qwen3.8-rocmi4/Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00001-of-000NN.gguf \
  --artifact /scratch/qwen3.8-rocmi4/Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00002-of-000NN.gguf \
  --mtp /scratch/qwen3.8-rocmi4/Qwen3.8-Flash-Next-MTP-ROCmI4-Strix-Halo.gguf \
  --mtp-sha256 <digest-from-the-sealed-final-bakeoff-evidence> \
  --mmproj /scratch/qwen3.8-rocmi4/Qwen3.8-Flash-Next-BF16-mmproj.gguf \
  --vision-vocab /scratch/qwen3.8-rocmi4/Qwen3.8-Flash-Next-vocab-only.gguf \
  --license /models/Qwen3.8-Flash-Next-source/LICENSE \
  --build-record /scratch/qwen3.8-rocmi4/qwen-quant-build-record.json \
  --engine-revision "$(git rev-parse HEAD)" \
  --container-image ghcr.io/otheru/ember@sha256:<64-hex-digest> \
  --out-dir release/qwen3.8-candidate
```

The stdlib-only program requires the text artifacts and completed record to come
from the same committed quantization directory. It copies every input through
a pinned file descriptor into a private sibling of `--out-dir`, rejects an
input whose inode, size, mtime, or ctime changes during the copy, and validates
only those stable copies. It writes the GGUF shards, selected MTP, `README.md`, `LICENSE`,
`artifact-manifest.json`, `SHA256SUMS`, `qwen-quant-build-record.json`,
the byte-exact applied `qwen-intervention-manifest.json`,
the separate BF16 mmproj, the matching zero-tensor vocab-only GGUF,
`release-profile.json`, and `upload-plan.json`, syncs them, then publishes the
entire previously absent output directory with
`renameat2(RENAME_NOREPLACE)`. Existing directories, files, and dangling
symlinks are never replaced. The package generator revalidates the preflight's exact
artifact bytes, ordered shard count/sizes, 128 GiB budget, 32 GiB reserve,
total, headroom, and passing fit result before it writes the candidate. Repeat
`--artifact` in order through shard `000NN`; the abbreviated two lines above
show the required convention rather than a fixed shard count. The plan records
absolute local inputs and Hugging Face destination paths. It contains no token,
has `publishes: false`, and has no network or upload implementation. Its
certification status remains `pending` for the non-layout release gates below.
The mmproj is copied through the same stable-file boundary and its GGUF header
must identify a `clip` Qwen3-VL merger with file type `MOSTLY_BF16`, projection
width 2560, spatial merge 2, and an unquantized BF16/F32 tensor inventory. Its
exact size, digest, metadata evidence, and pending real-weight certification
are recorded separately from the quantized text shard budget.
The vocab-only GGUF must contain zero tensors and the pinned Qwen architecture,
tokenizer, and 2560-wide embedding metadata required by the vision provider.
Its exact size, digest, and GGUF metadata digest are recorded independently.
It also requires the record's atomic-directory transaction evidence, tensor
inventory digest, and ROCmI4 type histogram, and binds the completed record to
the selected profile SHA-256, snapshot
inventory, source revision, llama.cpp/ROCmFPX/Ember revisions, quantizer build
identity, execution mode, and exact-dequant gate. Recorded output basename,
size, and hash must match every supplied artifact. For split releases the
manifest's singular compatibility field is an explicit aggregate shard-set
summary (`sha256: null`); authoritative per-file hashes remain in `artifacts`.
`SHA256SUMS` is the deployable runtime bundle boundary, not a recursive hash
of mutable release metadata. It is GNU `sha256sum` text with safe basenames
only, ordered as every selected main shard, the exact bakeoff-selected MTP,
then the exact BF16 mmproj, and finally the exact vocab-only GGUF. Missing,
reordered, duplicated, path-bearing, or otherwise unsafe names fail packaging.
`artifact-manifest.json` records the
checksum file's own SHA-256, exact ordered filenames, and entry count; the
publication envelope reproduces this list and binds all three flattened
companion artifacts (`mtp`, `vision_mmproj`, and `vision_vocab`) to the final
measured hardware evidence. The upload plan separately hashes every
release-evidence file, avoiding a checksum cycle through the manifest.
The generated model card states that this is an actual weight intervention and
lists both pinned intervention tools; it does not infer “Heretic” from a name.

## Release gates

1. Re-fetch the base snapshot by the exact commit, verify all expected files,
   the license digest, and the upstream weight-byte count (360,000,192,888).
2. Verify the intervention manifest byte-for-byte, including the pinned OtherU
   and Heretic revisions, corpus/direction hashes, zero held-out overlap, exact
   QSA/GDN residual-writer map, shapes, scales, and row normalization. Confirm
   both quantizer reports and every GGUF shard carry the same manifest and
   tensor-map hashes; a missing or zero applied-target count is a hard failure.
3. Build the converter and runtime from recorded commits in an immutable OCI
   image; retain command lines, logs, package versions, and artifact SHA-256.
4. Validate GGUF metadata, tensor inventory, dimensions, quant types, shard
   completeness, tokenizer/config parity, and a CPU-readable smoke path before
   scheduling scarce target hardware.
   Confirm the manifest's ordered shard sizes/hashes exactly match the completed
   aggregate preflight and immutable candidate revision.
5. Run the existing quant-quality pipeline against both the stock BF16 base and
   a pinned intervened BF16 reference. In addition to perplexity, vision,
   reasoning, tool-use, and long-context gates, publish held-out refusal bypass,
   benign over-refusal, agentic coherence, sampled optional-tool over-trigger,
   required-tool-use, and quantization-delta results. Direction-extraction
   prompts cannot appear in the held-out refusal set.
   Exact dequantization must pass its own perplexity, behavioral, vision,
   reasoning, tool-use, and long-context gates.
6. If W4A4 will be offered or benchmarked, repeat the quality suite and runtime
   certification with W4A4 explicitly enabled. Never substitute an exact-path
   report for this gate or merge the two results in a single headline number.
7. On the dedicated gfx1151 runner, verify cold load, representative prefill,
   autoregressive decode, context growth, memory high-water mark, determinism,
   and restart behavior. Use the exact candidate digest and disposable KV data.
   The differential validator retains `differential-decode-comparison.json`
   from its 64-token warm AR baseline, first restored MTP pass, and warm fresh
   MTP pass. This same-process comparison attributes whether speculation helps
   before another model load; it is diagnostic evidence and does not replace
   the separate three-sample unprofiled 2,074/256 hard gate. Both the matched
   Q3/IU4 comparator and the final candidate bakeoff independently derive its
   rates and speedup from the pinned 64-token durations; the hardware summary
   must exactly reproduce the pinned diagnostic.
   Qwen's embedded MTP path must not be described as a separate draft-model
   file unless the implementation actually requires one.
   Certification must explicitly exercise the PR #27742 bring-up hazards:
   sufficient `graph_max_nodes`, `ggml_argsort_top_k` above 1,023 candidates,
   and serialized indexer-cache state across snapshot restore.
   The current q=1 correctness accounting is 14,495,514,624 bytes for the exact
   native cache, 117,669,888 bytes for GDN state, plus an 8 GiB runtime reserve:
   23,203,119,104 bytes total with copy-on-write accounting. This is a
   correctness-first budget record, not a performance or full-process
   high-water claim; certification must still measure peak RSS/GTT.
8. Attach raw benchmark records and their workload definitions to the candidate
   revision. External SGLang, vLLM, or TokenSpeed measurements must preserve
   their exact prompt/output sizes, concurrency, batch, cache, precision,
   offload, image, model revision, and hardware settings and remain labeled
   `external comparison baseline`, not Ember results.
9. Review the generated model card, copied license, Qwen commercial-use terms,
   checksums, provenance, and quality reports. Legal review is an organizational
   gate; this document is not legal advice.
10. Upload first to the generated `candidate/<source>-<engine>` revision. Verify
   every remote file and digest by immutable candidate commit. Promote that
   exact commit only after approval; do not rebuild between certification and
   promotion.

## Runners and authentication

Quantization/package work belongs on a dedicated runner such as
`[self-hosted, linux, x64, qwen-quant]` with at least 1152 GiB free local storage
(1.5 TiB preferred), at least 256 GiB physical RAM, and a cache isolated from serving
models. Runtime certification belongs on the existing
`[self-hosted, linux, x64, gfx1151]` path. It must use the fixed production
quiesce/restore wrapper, exclusive GPU access, direct-I/O model-integrity cache,
and an exact artifact digest. The ordinary GPU-free CI runners only lint and
test the profile and packaging code.

For publication, prefer a Hugging Face Trusted Publisher bound to the exact
GitHub repository, environment, workflow, and target model, with
`permissions: id-token: write`. The workflow exchanges OIDC identity for a
short-lived credential, so no Hugging Face token is printed or stored. Creating
the target repository and Trusted Publisher is a one-time organization
contributor/admin action.

If OIDC is unavailable, use a fine-grained `HF_TOKEN` secret with write access
only to the target model repository. Pass it through the CI secret store to the
official client, never as an argument, URL, generated plan, log line, or file in
the workspace. Disable fork-triggered publishing and protect the release
environment. Local publishing requires separately installed Hugging Face CLI or
`huggingface_hub`, `git-lfs` where applicable, an authenticated account with
write permission, and enough disk for upload staging; none is required by the
offline generator.

## Hugging Face candidate and promotion workflow

The publishing workflow is deliberately separate from the generator and must consume
`upload-plan.json` without changing its paths or bytes:

1. An organization contributor creates the model repository once (private at
   first if policy requires) and configures its Trusted Publisher. The CI job
   receives only `id-token: write` and repository read permissions.
2. The official Hugging Face client creates the plan's exact
   `candidate/<source>-<engine>` branch from `main`, then uploads each planned
   file to that branch. No `latest` model, container tag, or mutable source
   revision is resolved inside the publishing job.
3. Record the candidate commit returned by the Hub. Resolve every uploaded file
   by that immutable commit, compare byte counts and SHA-256 values with the
   plan, and run target certification against the GGUF at that digest.
4. Open a Hub pull request from the candidate branch to `main`. Approval merges
   the already-reviewed bytes; it does not invoke conversion or regenerate the
   package. Record the resulting `main` commit and re-check all file hashes.
5. Make the repository public only after the model card, license, quality
   evidence, target benchmark, and checksums are visible at the verified
   `main` commit. Retain the candidate branch and both commit identifiers for
   audit and rollback.

The generated plan is safe to inspect or archive because it contains no
credential value. A future publisher should reject an unprotected branch, an
unexpected repository ID, a destination outside the plan, or any hash change
instead of trying to repair the candidate implicitly.

### Protected publication envelope

`qwen_release_package.py` remains offline and non-publishing. After the final
held-out confirmation, create a separate fail-closed handoff with
`scripts/qwen_publication_envelope.py build`. It requires explicit paths and
SHA-256 values for `upload-plan.json`, the final ledger and its GitHub artifact
attestation bundle, the complete measurement manifest, the audited quality
contract, the matching-MTP real-weight hardware record, and the immutable
runtime image. It also requires the exact `vision-certified.json` path,
SHA-256, and retained GitHub attestation bundle path/SHA-256 from the protected
gfx1151 vision workflow. The offline tool binds the bundle's one in-toto
subject digest to that evidence; the workflow verifies its Sigstore identity
against `OtherU-AI/ember/.github/workflows/qwen-gfx1151-vision.yml` immediately
after capture and again before publication OIDC, with both `source-digest` and
`signer-digest` fixed to the certified Ember revision. The tool replays the final 2,074-token prefill and 256-token
decode decision from the pinned evidence, checks the exact three-sample
performance/memory gates, binds the selected model inventory to every planned
package byte, and records the runtime image, Ember revision, engine-binary
digest, tensor-format contract, ordered model inventory, selected MTP, BF16
mmproj, provider/reference revision, cold/warm residency record, exact projected
rows, and pinned image-grounded visible-answer checks for both synthetic vision
cases. It uses only the Python standard library,
does not read credentials, and does not contact the Hub.

The resulting `ember.qwen3.8.hf-publication-envelope.v3` authorizes only its
generated `candidate/...` revision. It explicitly denies promotion. Verify it
again immediately before publication:

```bash
python3 scripts/qwen_publication_envelope.py verify \
  --envelope /protected/qwen/publication-envelope.json \
  --envelope-sha256 <sha256> \
  --expected-engine-revision <40-hex-ember-commit>
```

`.github/workflows/qwen-hf-candidate.yml` is the only provided Hub mutation
path. It is manual, runs only from `main`, uses the protected
`qwen-hf-candidate` GitHub environment, verifies the final-ledger GitHub
attestation before requesting OIDC, refuses an existing candidate branch,
uploads only the envelope's exact plan entries, resolves the branch to an
immutable Hub commit, downloads every planned file by that commit, rehashes
every byte, and attests the verification receipt. It contains no promotion job
or event. The certified engine revision must already be an ancestor of the
protected `main` workflow revision, so policy-only commits do not invalidate a
completed hardware record and an unmerged research revision cannot publish.

Before the workflow can run, an organization administrator must complete these
one-time controls:

- create the target model repository without populating the candidate branch;
- configure a Hugging Face repo Trusted Publisher for resource
  `otheru/Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-GGUF`, pinning repository
  `OtherU-AI/ember`, branch `main`, and workflow `qwen-hf-candidate.yml`;
- create the GitHub `qwen-hf-candidate` environment with required reviewers;
- install exactly `huggingface_hub==1.19.0` on the protected `qwen-quant`
  runner. The workflow deliberately does not download executable tooling after
  obtaining publication authority.

Hugging Face Trusted Publisher tokens are short-lived and repository-scoped.
The workflow exposes neither an `HF_TOKEN` secret nor a fallback credential.
If any one-time control is absent, OIDC exchange or the runner-version check
fails closed. Promotion must be implemented and approved separately, must name
the verified immutable candidate commit, and must rehash the resulting `main`
commit; this repository intentionally provides no automatic promotion path.
