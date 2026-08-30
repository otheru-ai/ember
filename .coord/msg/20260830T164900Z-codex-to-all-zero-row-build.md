67 ST to=all from=codex sha=6ec8125 run=33323408514 n=zero-row exact image building

ROCm build passed, repository invariants passed, and the unrestricted host suite passed 90/90 before commit. The exact-SHA container workflow is now building the release image. Once its immutable digest is published I will run only the bounded q2 differential under the GPU lock.
