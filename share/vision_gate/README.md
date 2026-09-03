# Vision behavioural gate — pre-registered inputs

The gate asks one question: does the model answer from the picture, or from
text priors? Arm A sends the image, arm B sends the same question with **no**
image, and any item arm B answers correctly is cut from arm A's score rather
than counted. A model that ignores images cannot pass by being a good guesser.

Everything here is pre-registered and digest-bound. `scripts/ds4_vision_behavior_gate.py`
carries the SHA-256 of each file as a constant and refuses to run against
anything else, so a threshold cannot be edited after seeing a number.

| file | SHA-256 | what it fixes |
| --- | --- | --- |
| `policy-v4.json` | `b82d86ff…` | thresholds, scoring, generation budget, the arm-B cut rule |
| `synthetic-manifest.json` | `f1b8303e…` | 100 generated items, 4 classes x 25, each image bound by digest |
| `natural-manifest.json` | `b69b1979…` | 100 TextVQA photographs, likewise |

The images themselves are not in git — the natural set alone is 95 MB. They
live on the certification box under the root named by
`EMBER_CERT_VISION_GATE_ROOT`, and each manifest entry carries its image's
SHA-256, so a substituted picture fails the run rather than changing a score.

## Why v4 and not v1

v1 capped the completion budget at 32 tokens. This model emits
`reasoning_content` before `content` and reasoning spends the same budget, so
every item returned `finish_reason: "length"` with empty content and the gate
went red for a harness reason with no bearing on the model. v4 raises it to 512
(2048 for the longer natural answers) and adds the rule for a degenerate
generation refused with HTTP 422: a scored miss in the no-image arm, where
degenerating *is* the measurement, and still an error with the image present,
where it is a real failure.

Both changes were made before any gate item had been scored, from an
independent run — recorded in the policy's own `max_tokens_note`.
