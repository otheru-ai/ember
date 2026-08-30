37 ST re=33 n=q2/q3/q6/q17 all diverge; q2 flips output1

Exact b3 image, same retained Q3 and FAST MTP, `--validate-tokens 2`:

- prompt_tokens=2: mismatch index 1, expected 830, actual 198
- prompt_tokens=3: mismatch index 0, expected 830, actual 4010
- prompt_tokens=6: mismatch index 0, expected 10459, actual 87
- prompt_tokens=17: mismatch index 0, expected 87, actual 830

Thus the smallest batched path (logical q=2, physical q=5) initially preserves
top-1 but changes enough hidden state to flip the following q=1 decode; q=3 and
larger already flip the seed token. Production wrapper is active and healthy;
the GPU lock is free. Evidence:
`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/prefill-width-sweep-b3b16e3-20260830T171100Z/`.
