# Qwen3.8-Flash-Next performance baseline

Ember's initial Qwen4Exp text runtime is a correctness baseline, not a
performance claim. It processes prefill one token at a time, orchestrates many
one-row ggml graphs from C++, performs recurrent GDN and QSA selection on the
CPU, and dequantizes selected routed-expert rows into float for scalar CPU dot
products. Record it under the implementation label `cpu_orchestrated_q1`.
`fused_hip` is reserved for a future measured implementation; its appearance in
the recorder's label table does not mean that path exists.

## Record a clean baseline

Use the same exclusive-device/quiesce procedure as `scripts/profile_gpu.sh`.
Start the Qwen server at its native context, obtain the host PID (for a Docker
container, `docker inspect -f '{{.State.Pid}}' NAME`), and use the SHA-256 that
the direct-I/O model-integrity step already verified:

```bash
python3 scripts/qwen4exp_baseline.py \
  --dry-run \
  --artifact-manifest /models/artifact-manifest.json \
  --server-artifact /home/ember/build-rocm/ember-dflash \
  --server-pid HOST_PID \
  --out reports/qwen4exp-baseline.json

python3 scripts/qwen4exp_baseline.py \
  --artifact-manifest /models/artifact-manifest.json \
  --server-artifact /home/ember/build-rocm/ember-dflash \
  --server-pid HOST_PID \
  --out reports/qwen4exp-baseline.json
```

The release manifest is the preferred provenance input. The recorder consumes
its complete ordered `artifacts` array, requires every shard beside the
manifest with the recorded size, and stores every digest/file identity plus the
aggregate byte count and manifest digest. Legacy single-file artifacts may use
`--model PATH --model-sha256 VERIFIED_DIGEST`; those options cannot be mixed
with a manifest.

The default request shapes toward 240,000 prompt tokens, generates 128 tokens,
and runs against a server configured with the native 262,144-token limit. Input
shape is necessarily approximate because the HTTP seam accepts text; the
response's `usage.prompt_tokens` is authoritative and is preserved beside the
backend's separate prefill/decode timings. The 22k-token reserve avoids turning
tokenizer/template variance into an accidental context overflow.

The record binds the measurement to every model shard's supplied, previously
verified digest plus file identity, hashes the exact `server_artifact` locally,
records the source revision, and samples host `VmRSS`/`VmHWM` and amdgpu
`mem_info_gtt_used` through the request. Missing GTT sysfs data is recorded as
`null`, never zero. Do not hash the ~94 GB model through the buffered page cache
immediately before loading it; certification's direct-I/O integrity result is
the input to the release artifact manifest (or legacy `--model-sha256`).

Real recording currently rejects `--implementation-path fused_hip`: the
correctness runtime cannot authoritatively report such a path, so accepting the
operator's label would allow a CPU-orchestrated result to masquerade as fused.
Dry-run may show that future label for planning only. Lift the rejection only
after the response carries an authoritative execution-path field and the
recorder verifies an exact match.

Profiler runs are separate experiments. Use `scripts/profile_gpu.sh` for
rocprof timing/counters, because counter collection serializes dispatches.
Magpie is not installed or validated in this repository. If a future run
captures a supported torch trace through another serving stack, TraceLens's
inference stage convention is `prefilldecode`, `decode`, and `prefill`, and its
compact roofline CSVs should be reviewed by total kernel time before mapping a
hotspot to source. That workflow must not be presented as an Ember measurement
until the local Magpie compatibility matrix includes the actual ROCm/gfx1151
setup and the commands have run successfully.

The pinned `wmma-ops` gfx1151 source audit is in
[`docs/wmma-ops-c319068-audit.md`](wmma-ops-c319068-audit.md). Its large square
FP16 and prepacked W4A4 measurements are experiment inputs, not Qwen baseline
results: the current runtime is q=1, its weights/layouts differ, and the remote
repository's published shapes differ. OtherU ownership has been clarified by
the user for internal evaluation; an explicit repository license remains a
public-redistribution prerequisite. Use its paired-K schedule only as a future
comparison for the already-separate opt-in W4A4 path, after profiling shows
that path is hot.

## Estimated bottlenecks and kernelization order

The figures below are static source-level estimates, not measurements.

1. **Routed experts.** Each token selects 10 experts in each of 48 layers. One
   expert evaluates `2560x1280 + 640x2560 = 4,915,200` weights, so the routed
   path performs about 2.36 billion MACs (4.72 GFLOP at two operations/MAC) and
   has an ideal 4-bit payload floor of about 1.18 GB per token, before scales
   and metadata. Today every output row is expanded to float and dotted by a
   scalar CPU loop. First fuse ROCMI4 dequantization, selected-expert matvec,
   activation, and weighted accumulation without materializing float weights.

2. **Per-matvec graph construction and synchronization.** The non-rotation
   path constructs roughly 783 single-operation ggml graphs per token. If the
   optional 256x256 cache rotations are present, their per-head loop can add
   roughly 624 more. Each call allocates a context/graph/allocator and copies
   its output back to the host. Replace this with persistent buffers and a
   stable q=1 graph, then fuse HC normalization/projections where profiling
   shows launch-bound regions.

3. **GDN recurrence.** Thirty-six layers update `48x128x128` float recurrent
   states. The live state alone is about 113 MiB across those layers, and the
   scalar loops repeatedly decay, dot, rank-one update, and query it. A fused
   GPU recurrence kernel removes CPU serialization and host/device barriers.

4. **Native-depth QSA.** At 262,144 tokens, each of 12 QSA layers scans 65,536
   four-token index blocks on the CPU before selecting 512. Reading each raw
   128-float index key once is about 128 MiB per layer, or 1.5 GiB per decode
   token across QSA layers, before scoring. The current per-query-head attention
   loops can reread roughly another 1.2 GiB of selected float K/V per token.
   Move pooling, RMSNorm/RoPE, top-k, and attention into one device pipeline
   that reuses KV across the 12 query heads sharing each KV head. At native
   depth this may outrank earlier items; let the clean record and rocprof total
   contribution decide.

5. **Static tensor transfers and PLE.** Norm/conv tensors are copied and
   converted repeatedly even though weights are immutable. Cache converted
   small tensors at load time. PLE's 16 random 160-wide rows should follow only
   if counters show it matters.

After every isolated kernel change, keep numerical parity with the q=1 path,
then rerun the same unprofiled record with identical model digest, prompt shape,
and implementation label changed only when the execution path truly changes.
