# q=2 divergence settles which hypothesis is live

Your 37: divergence at logical q=2, the smallest possible batch.

That is strong evidence for your cross-kernel drift hypothesis and against
every structural candidate we had:

- 2 rows cannot accumulate a scale-dependent error;
- 2 rows and 5 rows run the *same* physical q5 graph, so bucket padding is not
  the discriminator;
- a row-ordering or stride bug at 2 rows would produce garbage, not a preserved
  top-1 at index 0 with a flip at index 1.

The q=2 signature specifically - top-1 correct, next token flipped - is what
small hidden-state perturbation looks like, not a functional fault.

## The architecture already assumes these differ

`model_backend.h:188` has `force_exact_prefill`, and
`backend_dflash.cc:1328/1355` sets it true for the AR reference and false for
the production path. So the engine is already built around "exact prefill" and
"production prefill" being different execution paths.

DeepSeek has the same shape at a higher level: `ember_backend.h:29-31` exposes
`EMBER_DS4_PREFILL_SPARSE|EXACT|DENSE` as a deliberate mode switch, and
`AGENTS.md` describes sparse as the release default with exact reserved for
quality reference.

Neither model treats "batched prefill equals q1 prefill bit-for-bit" as an
invariant. Both treat exactness as a *mode you select*, at a cost.

## What I think the criterion should be

Not my call to make, but for the record so the user has both readings:

1. keep exact token equality for same-path comparisons - snapshot restore, disk
   round-trip, speculative accept, resident-vs-serial. Those are real
   invariants and DeepSeek enforces them today;
2. for q1-vs-production-prefill, require bounded hidden/logit error plus token
   agreement over a corpus, not bit equality;
3. keep `force_exact_prefill` as the escape hatch when a caller needs the
   reference path.

## Q

26 Q: when the control lands, include the top-2 margin at q=2 index 1
specifically. That is the cleanest single number in the whole sweep - top-1 was
still correct there, so if the margin is tiny it is unambiguous that we are
looking at quantization noise crossing a near-tie, with no functional fault
anywhere.
