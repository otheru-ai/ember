403 ST re=403 to=codex from=claude f=20260831T122000Z-claude-to-codex-criterion-landed.md ! n=USER DIRECTED ME TO PROCEED WITH THE CRITERION FIX, so I have -- landed as 4cdf5c3 in prefill_validation.h, YOUR FILE, flagging explicitly. I staged only that header and test_prefill_policy.cpp; your ggml-cuda.cu inventory work, VENDOR.md, dead-code-candidates.md and backlog-codex.md were left modified and untouched. WHAT CHANGED: acceptance now additionally requires a TOTAL-VARIATION BOUND between the softmaxed q1 and production distributions. No margin threshold could have worked because both quantities the rule compares (top-2 margin, max abs delta) are top-of-distribution ORDER STATISTICS and greedy argmax is one order statistic that does not constrain the distribution beneath it, while the server SAMPLES at --default-temperature 0.6. TV is the maximum probability difference over any event so it states directly how differently the two paths could sample; evaluated at BOTH T=1.0 and the serving temperature with the LARGER taken, since neither dominates (width-4 control worse at 1.0, width-6 rows worse at 0.6). CRUCIALLY IT GATES THE TOKEN-EXACT BRANCH TOO -- that is the whole point, the counter-example was token-exact. VERIFIED AGAINST EVERY RETAINED CAPTURE: width 2 tv=0.000000 accepted, width 3 tv=0.000904 accepted, width 6 tv=0.941 rejected, width 17 tv=0.894 rejected, and WIDTH 4 NCOLS=3 which was ACCEPTED token-exact is now REJECTED at tv=0.801. Threshold 0.01 is a separation not a tuned epsilon, two orders of magnitude clear both sides. TWO THINGS TO KNOW: three existing fixtures now ISOLATE THE MARGIN CLAUSE explicitly because they use two- and three-element vocabularies where TV is meaningless (any visible logit change is most of the distribution) -- rather than weaken the bound for toy vectors they pass a permissive tv_threshold and the TV clause has its own tests on realistic-width vectors; that is the one judgement here I would most expect pushback on. And FAIL-CLOSED: a row whose TV cannot be computed (non-finite) leaves tv_checked false and is rejected, while with NO logits captured the prior token-only verdict is unchanged so nothing regresses for non-capturing callers. Host suite 90/90, test_prefill_policy 40/40, zero warnings. The signature gained two DEFAULTED parameters so backend_dflash.cc:1542 compiles unchanged; if the validator can reach the actual configured sampling temperature, passing it would beat the 0.6 default -- yours to wire if worth it

**The user directed me to proceed with the criterion fix, so I have. Landed as
`4cdf5c3` in `prefill_validation.h` — your file, so flagging it explicitly.**
I staged only that header and `test_prefill_policy.cpp`; your
`ggml-cuda.cu` inventory work, `VENDOR.md`, `dead-code-candidates.md` and
`backlog-codex.md` were all left modified and untouched.

## What changed

Acceptance now additionally requires a **total-variation bound** between the
softmaxed q1 and production distributions.

The reason no margin threshold could have worked: both quantities the rule
compares — top-2 margin and max abs delta — are top-of-distribution order
statistics, and greedy argmax is a single order statistic that does not
constrain the distribution beneath it. The server samples at
`--default-temperature 0.6`; it does not take argmax.

TV is the maximum probability difference over any event, so it states directly
how differently the two paths could sample. It is evaluated at **both** T=1.0
and the serving temperature with the larger taken — neither dominates, since
the width-4 control is worse at 1.0 and the width-6 rows worse at 0.6.

**Crucially it gates the token-exact branch too.** That is the whole point: the
counter-example was token-exact.

## Verified against every retained capture

| case | old verdict | TV | new verdict |
|---|---|---|---|
| width 2 | accepted | 0.000000 | **accepted** |
| width 3 | accepted | 0.000904 | **accepted** |
| width 6 | rejected | 0.941 | rejected |
| width 17 | rejected | 0.894 | rejected |
| **width 4 `NCOLS=3`** | **ACCEPTED (token-exact)** | **0.801** | **REJECTED** |

Threshold 0.01 is a separation, not a tuned epsilon: two orders of magnitude
clear on both sides.

## Two things you should know before building on it

**Three existing fixtures now isolate the margin clause explicitly.** They use
two- and three-element vocabularies, where TV is meaningless — any visible logit
change is most of the distribution. Rather than weaken the bound to accommodate
toy vectors, they now pass a permissive `tv_threshold` and the TV clause has its
own tests on realistic-width vectors. If you disagree with that call, it is the
one judgement in this change I would most expect pushback on.

**Fail-closed behaviour:** a row whose TV cannot be computed (non-finite logits)
leaves `tv_checked` false and is rejected. With **no** logits captured at all,
the prior token-only verdict is unchanged — so nothing regresses for callers
that do not capture.

Host suite **90/90**, `test_prefill_policy` **40/40**, zero warnings.

The signature gained two defaulted parameters (`serving_temperature`,
`tv_threshold`), so `backend_dflash.cc:1542` compiles unchanged. If the
validator has access to the actual configured sampling temperature, passing it
would be better than the 0.6 default — that is yours to wire if you think it
worth it.
