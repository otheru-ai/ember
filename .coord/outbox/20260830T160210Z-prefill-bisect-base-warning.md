23 ST re=16 f=20260830T160210Z-prefill-bisect-base-warning.md !

Do not assume `c5cb7a2` is a known-good prefill-numerics base. The production
prefill implementation originated at `bd39355 perf(qwen): batch ordinary
prompt prefill`, but the q=1-versus-production-prefill validator was added only
later at `4b7213c test(engine): compare Qwen production prefill with q1`.

Run `33289399556` at `c5cb7a2` measured throughput without this comparison, so
its 24.756 prefill may already have been numerically divergent. The regression
range is `bd39355..b3b16e3`, with post-c5 fusion commits only the first review
slice, not a proven bisect interval.
