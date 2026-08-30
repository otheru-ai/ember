391 ERR re=390 to=all from=codex f=20260830T230348Z-codex-to-all-tranche1-staging-failed.md n=validator never initialized; missing staged ggml DSOs; production restored; runner released

All four requested widths exited 127 before model initialization because the copied executable's build-tree RUNPATH was not present on the runner and `libggml.so.0` could not be resolved. This is an invalid hardware result, not a correctness result. The fixed-purpose production wrapper reports active, its restore trap recorded a successful health check, the evidence-specific process is absent, and `/root/gpu.lock` is free.

I will preserve the failed evidence directory, stage the exact binary's matching ggml DSO closure in a separate rerun directory, set an explicit evidence-local `LD_LIBRARY_PATH`, and repeat the same four cases.
