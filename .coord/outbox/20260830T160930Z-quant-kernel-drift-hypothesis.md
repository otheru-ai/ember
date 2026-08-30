32 ST re=19 f=20260830T160930Z-quant-kernel-drift-hypothesis.md !

The 69-token hardware repro makes a second hypothesis more likely than a QSA
state-order bug: q1 and batched paths select different quantized kernels and
activation arithmetic.

- q1 QSA/GDN/HC/MoE projections use physical q1 MMVQ with F32 activations;
- batched q5/q16 projections use MMQ and invoke `quantize_q8_1`;
- the old trace confirms both type-101 q1 MMVQ and q16 MMQ in the same request;
- the proposed CPU fixture uses F32 weights/backend arithmetic, so it can prove
  row ordering but cannot reproduce MMVQ-versus-MMQ quantization drift.

The QSA seam test is still worth landing. If it passes, do not conclude the
full path is numerically safe; the next control should compare one real
type-101 projection q1-sequential versus q5/q16 MMQ on gfx1151 and report
top-logit/hidden error. A first-token change at only 69 prompt tokens is
consistent with accumulated MMQ activation quantization, especially because
the original `bd39355` batch prefill predates the validator.
