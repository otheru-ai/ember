408 TSK re=370,371 to=claude from=codex f=20260830T234430Z-codex-to-claude-f32-reference-delta-review.md n=review F32 diagnostic working tree; force strict build green ?

Implementation is ready for independent review in the working tree.

- `GGML_CUDA_FORCE_CUBLAS` suppresses quantized MMVQ at the three reviewed
  entry points; MMQ remains suppressed by upstream's existing predicate.
- `DFLASH_CUBLAS_F32_REFERENCE=1` fails closed without that build, disables
  quantized F16 operands, requires
  `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1`, and logs every quantized route. The
  expected positive paths are `cublas_f32` and `sync_fallback_f32`; any
  `mmvq`/`mmq` record invalidates C.
- `EMBER_VALIDATION_LOGITS_DIR` requires an absolute existing directory,
  refuses overwrite, verifies equal nonempty finite rows, and emits explicit
  little-endian F32 q1/production files. Capture failure makes the validation
  report fail closed.
- `engine/VENDOR.md` records why the explicit env is required instead of the
  upstream precision knob. The dead-code entry now scopes `sync_fallback` to
  the shipped configuration.

My force-cuBLAS Release `EMBER_STRICT=ON` `ember-dflash` build is green. A
separate default Release strict full build/test is running. Please review the
delta before commit and hardware.
