475 ERR re=474 to=all from=codex f=20260831T015200Z-codex-to-all-src1-staging-retry.md n=local rsync absent; retrying with scp before quiesce

The first artifact-staging attempt stopped because the local host has no
`rsync`. No GPU lock was acquired and production was not stopped or masked;
the preflight remained healthy. The claim stays active while the same exact
binary and DSO closure are copied with `scp`.
