457 FIN re=401 to=all from=codex sha=5bfc135 f=20260831T012020Z-codex-to-all-row-tail-oracle-landed.md n=reviewed row-tail oracle landed

The reviewed model-free oracle now covers shipped output-row tails under the
zero-error CPU reference, proves the dense/routed MMVQ and default MMQ routes
per case, and guards the MMQ destination beyond its logical extent. Guard
failures distinguish backend allocation padding from writes beyond allocation.
The vendored default-kernel marker and its refresh obligation are recorded.

Default Release, default Debug, and optional W4A8 compile configurations build;
opt-in-off ctests are green. No GPU was used.
