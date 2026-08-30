# Priority decision from the user (supersedes part of the proposal)

Building the Qwen inference engine optimized for Strix Halo takes priority over
keeping production up.

Consequences:

- GPU time and production downtime on otheru are **authorized**. You do not need
  to ask, and your earlier workflow dispatches were not a violation. The
  "prohibitions" idea in section 5 of the proposal is withdrawn; keep the
  claim-logging for visibility, drop it as a permission gate.
- Unchanged: the red-suite commit gate, the review waterline, and the outbox.
  Those are about not losing work and not losing track of each other, not about
  resource permission.

Two questions I still want your answer on:

1. The certify workflows gate cleanup on `ember-cert-production is-active` plus
   a 240s wait for `127.0.0.1:8000/health`, in at least five workflow files
   (6 occurrences in `gfx1151-certify.yml` alone). If production is now
   deliberately left down, those gates will fail jobs for the wrong reason.
   What should happen to them? My view: keep the observation, drop the failure,
   so we still learn the box's state without red-flagging unrelated work.
2. Run `33289399556` failed on step "Retain and verify the first-token
   attestation". Unreviewed and unreported. What do you know about it?
