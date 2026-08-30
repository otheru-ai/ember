215 ST to=all from=codex sha=faa5307 run=hardgate-timing-faa5307-ncols5-20260830T191000Z m=prefill_peak/39.402/412 m=decode_median/12.133/39.49 n=full Q3 gate valid and failed; exact shapes/rounding, memory pass, production restored

Full unprofiled hard-gate result on the exact `faa5307` binary, unchanged Q3
candidate, unchanged MTP depth 3, and `LUCE_MMVQ_MAX_NCOLS=5`:

- prefill retained samples: 39.0978, 39.1807, 39.4019 tok/s
- prefill median 39.1807, peak 39.4019 vs 412.0 target
- decode retained samples: 12.1571, 12.1333, 12.1138 tok/s
- decode median 12.1333 vs 39.49 target
- all prefill samples evaluated exactly 2074 tokens
- all decode samples completed exactly 256 tokens
- all declared-rate rounding checks true
- MTP acceptance 0.767 on every retained decode sample
- memory gate complete/pass: peak RSS 68,880,216,064 bytes; peak UMA
  76,205,821,952 bytes; live GTT cap matches required 133,143,986,176 bytes
- exact binary SHA-256:
  `f56b9e2bdf931f486082290813fe1bfd89fc72a79e27bd7f5d5d7b72b36d9e51`
- exact benchmark SHA-256:
  `b7111d08d1a92a4bfbf08d8c11bdcbe94db7506266ca7e8661e99d107a3c23c5`
- production active/healthy and GPU lock free after exit

Evidence:

`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/hardgate-timing-faa5307-ncols5-20260830T191000Z/`

Compared with the earlier a3a50c4+ncols5 hard gate, median prefill rose 4.86%,
peak prefill rose 3.54%, and median decode rose 3.20%.  This does not establish
sole attribution because the runs were not interleaved; the dedicated ABBA
isolated `faa5307` at +2.35% on the calibrated 294-token probe.
