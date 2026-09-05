# Runtime directional steering

Ember can project an operator-supplied direction out of each target layer's
attention and FFN outputs. It is disabled unless configured at startup. This
is an experimental model intervention; its effect and cost require measurement
on the selected checkpoint.

```sh
ember-dflash -m /models/target.gguf \
  --dir-steering-file /models/direction.f32 \
  --dir-steering-attn 0 --dir-steering-ffn 3.5
```

The file contains exactly 704512 bytes: 43 layers by 4096 channels, layer-major,
little-endian IEEE float32, with no header. Supply directions derived from the
target checkpoint. Values are used verbatim, without normalization. An all-zero
row excludes that layer. Nonfinite values, short or long files, and missing
files fail startup, including when both scales are zero. Scales must be finite
and in `[-100,100]`. A nonzero scale requires a file.

With a file and neither scale flag, FFN defaults to 1 and attention to 0,
matching ds4's CLI. Specifying either scale leaves the other at zero; an
explicit zero is respected. Supplying no flags preserves existing behavior.

For each token, each enabled sublayer computes

```text
y' = y - scale * v * dot(v, y)
```

Attention is projected after its final output projection. FFN is projected
after the shared and routed expert contributions have been combined. Both
operations precede HC post, so this is an intervention on the sublayer output,
not on the mixed residual streams. The operation follows ds4.c's
`cpu_directional_steering_project_rows` contract. The local engine additions
are Ember fork divergence.

The target's exact and approximate prefill, fused AR decode, layer-range
verification, and mixed image/text FFN paths share these projection sites.
Hybrid expert placement projects the aggregate FFN on the CPU. Zero scales
and excluded layers add no graph nodes or floating-point operations.

DSpark's proposal layers correspond to target indices 43, 44, and 45, outside
the direction file. Its proposal weights receive no target direction rows.
Target verification and the target features captured for DSpark do receive
steering; target acceptance remains authoritative. The cost and acceptance
rate of those proposals must be measured. The 43-row policy does not support
the diagnostic single-layer target slice.

Directions are read once and retained for the backend's lifetime. Device
tensors and the disk KV identity use that same immutable content. The identity
includes the effective scales and every direction value, extending Ember's
existing model file identity salt. Changing content at the same filename or
changing scales creates a different disk namespace. Disabled policies retain
the existing namespace. Reload the backend to change the intervention.

## Validation and release gate

The host suite exercises file validation, CLI startup, cache identity and the
CPU projection. `test_ds4_steering_graph` executes the production projection
helper on real ggml CPU graphs at widths 1, 2, 4, 17 and 64, for attention and
FFN with scales 0, 1, -1 and 3.5. These are arithmetic and structural checks;
they do not establish model quality or GPU batch parity.

Before enabling the feature in a deployment, the GPU owner must verify:

- An unchanged default-off baseline, followed by AR/snapshot/disk/speculative
  differential validation using the exact candidate binary, model, direction
  digest and scales. Include resident sessions and native vision where used.
- A positive behavioral control at the previously specified scale 3.5. A
  synthetic projection test cannot show that the direction affects refusals in
  the actual model; that control must visibly respond before a lower-scale
  negative result is interpreted.
- The predeclared repetition and tool-quality evaluation at the intended
  serving scale, with matched prompts, samplers, token budgets and cache state.
- Matched throughput, prefill time, memory and DSpark acceptance measurements.
  No overhead percentage is established by the implementation or host tests.

No hardware, behavioral, or performance result is implied by a successful
container compile. Deployment remains gated on independent review and those
measurements.
