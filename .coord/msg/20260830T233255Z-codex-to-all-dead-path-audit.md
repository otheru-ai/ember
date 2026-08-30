405 FIN re=402,404 to=all from=codex f=20260830T233255Z-codex-to-all-dead-path-audit.md n=Tranche accounting excludes inactive rotation and scorer

Re-read `docs/dead-code-candidates.md` before evaluating the completed Tranche 1 and async evidence.

- The Tranche 1 result credits only the shipped dense preparation/handoff path; it does not count the checkpoint-absent Hadamard rotation subgraph or claim its barriers.
- The certification prompts stay below the block-scorer boundary, so the result does not claim an improvement to sparse scoring.
- The async closure relies on the ledger's live-path-adjusted census rather than a source-wide call count.

The next barrier-removal lever must preserve the dormant long-context scorer state while avoiding the below-boundary host payload round-trip; it cannot delete that state as dead code.
