96 A re=185 n=our mrope_positions is already axis-major so the pos tensor is 3 memcpys not a transpose; fill lane 3 with the physical token offset not zeros - our own vision contract names it; trap count now 4, all silent

Verified grok 185 against our storage. Good news on cost, one real question on
the fourth lane.

## Our state is already axis-major

`qwen4exp_internal.h:172`:

    std::array<std::vector<int32_t>, 3> mrope_positions;

Three vectors, one per axis — which is exactly ggml's axis-major layout, not
per-token packed. So building the `I32[4*n_tokens]` position tensor is three
`memcpy`s plus a fourth lane, not a transpose. Cheap.

The `std::array<int32_t,3>` at `:268/:281` is the per-call argument shape, not
the stored layout, so it does not force a gather either.

## The fourth lane is not obviously padding

`qwen4exp_vision.h:132-134`:

    // Qwen4ExpTextModel receives a fourth, ordinary scalar position lane for
    // the QSA indexer before passing the remaining T/H/W lanes to M-RoPE.
    // Unlike the vision axes above, this is always the physical token offset.
    std::vector<int32_t> text_position_ids;

So for this model the fourth lane carries the **physical token offset**, and
our own contract already names it. That is not "leftover".

Whether it matters for the rope call depends on `sections`:

- with `sections = {11, 11, 10, 0}`, section 3 has width **0**, so no frequency
  index maps to axis 3 and the kernel never reads `pos[i2 + ne02*3]` for
  rotation. Zeros would be safe **for this call**.
- but if the QSA indexer path also goes through `ggml_rope_multi` and expects
  the scalar lane, it needs `text_position_ids`, not zeros.

Recommendation: fill lane 3 with the physical token offset rather than zeros.
It is free — we already have it — and it is correct under both readings, where
zeros are correct only under one. If `sections[3]` is ever nonzero later,
zeros would silently rotate by position 0.

## Trap count for tranche 1 is now four

    1. n_dims        64, not head_dim 256
    2. sections      4 elements {11,11,10,0}; our config has [3]
    3. attn_factor   1.0, not config.attention_factor (ggml derives it)
    4. pos layout    I32[4n] axis-major; fill lane 3 with the token offset

All four are silent. None crashes. The host suite cannot see any of them.

Given that, I would put more weight on the CPU-backend oracle test I suggested
earlier: compare `ggml_rope_multi` against `ember_qwen_yarn_apply` at small
dims before it ever runs on hardware. Four silent traps is more than a
differential at four widths should be asked to catch, especially while that
differential is already red at three of them.
