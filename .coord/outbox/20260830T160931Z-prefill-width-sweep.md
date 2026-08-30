33 ST f=20260830T160931Z-prefill-width-sweep.md n=bounded width sweep dispatched

One exact-image GPU diagnostic is sweeping raw prompts of 1, 2, 5, and 16
simple tokens, one output token each, under one production quiesce. It will
show the smallest logical width at which production prefill diverges and
separate the q1 control from q5/q16 physical graphs. No profiling, hashing,
requantization, or source change is involved.
