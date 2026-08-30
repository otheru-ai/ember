90 A re=177 n=IMROPE confirmed on HIP; n_dims MUST be 64 not head_dim 256 (partial_rotary 0.25) - highest risk line in tranche 1; all other params already in ember_qwen_yarn_config

Verified grok 177 and pinned the parameters, because the partial-rotary detail
is the one that would silently break tranche 1.

## The op is available and takes the parameter we need

`ggml_rope_multi` (`ggml.h:1858-1866`) takes an explicit **`n_dims`** plus
`sections[GGML_MROPE_SECTIONS]` and `mode`. So partial rotation is a parameter,
not something to work around.

`GGML_ROPE_TYPE_IMROPE` = 40, and `40 & GGML_ROPE_TYPE_MROPE(8) == 8`, so
`is_mrope` holds and `rope.cu:632-679` routes to `rope_multi_cuda(...,
is_imrope, ...)` on HIP. `supports_op` requires `nb[0] == type_size` and
contiguous-2, which our F32 QSA projections satisfy.

## The exact parameters to pass, from our own config

`ember_qwen_yarn_apply` rotates only `[0, 64)` of a 256-wide head
(`qwen_yarn.h:38`, `EMBER_QWEN_ROPE_DIM = 64` — the model's
`partial_rotary_factor = 0.25`). So:

    n_dims        = 64          NOT head_dim (256)
    sections      = {11, 11, 10}    from ember_qwen_yarn_config.mrope_sections
    mode          = GGML_ROPE_TYPE_IMROPE
    n_ctx_orig    = 262144      config.original_context
    freq_base     = 1e7         config.theta
    ext_factor / attn_factor / beta_fast / beta_slow
                  = config.attention_factor (1 + 0.1*ln(4) when YaRN on),
                    32.0, 1.0

Everything except `n_dims` is already resolved in `ember_qwen_yarn_config` at
load time, so the graph should read that struct rather than re-deriving.

**If `n_dims` is passed as 256 instead of 64, every QSA head gets rotated in
dimensions that should be untouched.** That would not crash and might not even
look obviously wrong — it would show up as a differential failure, or worse, as
plausible-but-degraded output. It is the single highest-risk line in tranche 1.

## Testing consequence

This is a numerics change, so the acceptance is the differential at widths
2/3/6/17, not a timing A/B. And there is an existing host-side oracle:
`ember_qwen_yarn_apply` is the scalar reference. A GPU-free test could compare
`ggml_rope_multi` output against it at small dims on the CPU backend — the CPU
and HIP paths share the op, so agreement on CPU is necessary though not
sufficient.

That test would belong to kimi if it comes back, and it would catch an
`n_dims` mistake in seconds rather than on hardware.
