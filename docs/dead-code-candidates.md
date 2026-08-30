# Dead-code candidates

Engine code that cannot execute on the configuration we ship — gfx1151 / RDNA
3.5, the published Qwen3.8-Flash-Next checkpoint, F32 KV cache — recorded here
rather than deleted on sight.

**This is a register, not a changelog.** Deleting is a separate decision with an
owner. Some entries are portability code that is correct to keep; the point of
writing them down is that "this never runs" stops being rediscovered.

Every entry states:

- **Evidence** — what was read or measured, with file:line.
- **Falsifier** — the check that would prove it is *not* dead. If the falsifier
  has been run, say so and by whom.
- **Scope of deadness** — architecture, checkpoint, or configuration. These are
  very different: an arch-dead path stays dead, a checkpoint-dead path revives
  the day someone ships different weights.
- **Recommendation** — delete, keep-with-comment, or undecided.

Vendored `engine/ggml/` and `engine/dflash/deepseek4/` are upstream's. Record
findings there, but do not delete: see [`engine/VENDOR.md`](../engine/VENDOR.md).

---

## 1. QSA Hadamard KV-cache rotation — dead on the published checkpoint

**Scope:** checkpoint + configuration. Not architecture.

The llama.cpp PR #27774 optional KV-cache rotation: `self_k_rot` applied to Q/K,
`self_v_rot` to V and to the attention output.

**Evidence.** The loader binds these only from GGUF tensors `attn_k_rot.weight`
and `attn_v_rot.weight` (`engine/dflash/qwen4exp/qwen4exp_loader.cpp:156-157`).
The pinned upstream snapshot `Qwen/Qwen3.8-Flash-Next` `f5d08274`
`model.safetensors.index.json` holds **1658 tensors and zero** whose name
contains `rot`, `k_rot`, `v_rot` or `hadamard` (grok, `.coord/msg/`
`20260830T191200Z-grok-to-codex-no-k-rot.md`). The GGUF is quantized from those
safetensors, so a tensor absent upstream cannot appear downstream.

With both null: `qsa_create_q1` skips the whole rotation subgraph
(`qwen4exp_frontier.cpp:1419`), and `qsa_rotate_q1` returns `true` at `:1535`
before touching a buffer. `qwen4exp_model.h:20` already records the intent —
the F32 cache "needs no rotation".

**Surface.** ~58 references to `rotation` / `rotated_` in
`qwen4exp_frontier.cpp`, 3 in its header, plus `rotate_optional`
(`qwen4exp_runtime.cpp:83`, called at `:580`, `:581`, `:633`, `:717`) and
`rotate_optional_batch` (`:1437`, called at `:1494`, `:1496`).

**Consequence for measurement.** Two of the fourteen host barriers in the
census — `qwen4exp_frontier.cpp:1550` and `:1561` — never execute. Over 12 QSA
layers that is 24 barriers and 48 copies per token that no A/B may credit any
change with removing. The live census is **12 barriers / 26 copies**.

**Falsifier.** Run twice, negative both times. Grok checked the upstream
safetensors index (above). Codex then checked the **GGUF shard-1 header
directly** — the file the loader actually opens — and reports
`attn_k_rot_count = 0`, `attn_v_rot_count = 0` (`.coord/msg/`, codex 204). That
closes the inference chain at the artifact rather than at its source. A
checkpoint that ships either tensor revives every line above.

**Recommendation: keep, comment.** This is upstream-parity code for a
configuration Qwen may yet publish, and it is already correctly gated — it
costs one null check per call, not a copy. What it must not do is keep
appearing in performance accounting as if it ran. Comment the gate at
`:1419` and `:1535` with the tensor names and this file.

---

## 2. `ggml_flash_attn_ext_set_prec(GGML_PREC_F32)` — inert on gfx1151

**Scope:** architecture, conditional on a build flag.

`qwen4exp_frontier.cpp:1307` requests F32 precision on the QSA attention node.

**Evidence.** Across the whole HIP backend, `ggml_flash_attn_ext_get_prec` is
read at exactly one site: `engine/ggml/src/ggml-cuda/fattn-wmma-f16.cu:562`.
The WMMA FA kernel is not selected on gfx1151 unless the build defines
`GGML_HIP_ROCWMMA_FATTN` (`fattn-wmma-f16.cuh:18-19`), which our build does not.
The TILE and VEC kernels that actually run never read the field. `fattn.cu`
contains no reference to `prec` at all.

So the call sets an op_param nothing consumes, and the attention runs at
whatever precision TILE/VEC pick regardless.

**Falsifier.** Build with `GGML_HIP_ROCWMMA_FATTN` and the call becomes live;
so does any A/B that compares against it.

**Recommendation: keep, comment.** Deleting it would silently change behaviour
the day rocWMMA FA is enabled. But it should not be cited as evidence that QSA
attention runs in F32 on this hardware — it is not.

---

## 3. `sync_fallback` in `ggml_cuda_mul_mat_id` — never taken

**Scope:** configuration. Vendored code.

`engine/ggml/src/ggml-cuda/ggml-cuda.cu:2648-2762`. The path performs two
`cudaStreamSynchronize` calls and a host-side expert loop.

**Evidence.** Measured **0 of 4924** MoE dispatches took it. It was investigated
as a suspected source of per-token hard syncs and refuted.

**Recommendation: no action.** Vendored; upstream needs it for cases we do not
hit. Recorded so the next person does not re-derive it as a lever. See
`docs/qwen3.8-performance-status.md`.
