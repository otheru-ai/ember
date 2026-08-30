27 ST f=20260830T160437Z-short-prefill-gpu-repro.md n=bounded gpu repro dispatched

I am running one manual exact-image diagnostic against `b3b16e3`: four short
constraint lines, two validation tokens, no disk cache and no profiling. It
uses the retained Q3 + FAST MTP bytes, takes the standard GPU lock, preserves
production's observed active state, and restores health on every exit. Purpose:
determine whether the q1-vs-batched mismatch reproduces cheaply enough to use
as the hardware confirmation loop after the host test narrows a fix.
