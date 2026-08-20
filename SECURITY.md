# Security policy

Please do not disclose vulnerabilities in an issue. Use the repository
host's private security-advisory feature. If that feature is unavailable, email
`otheru@otheru.ai` with `Ember security` in the subject. Include the affected
revision, reproduction steps, impact, and any proposed mitigation. Do not attach
model weights, private prompts, credentials, or operational data.

Only the latest release and the current `main` branch receive security fixes.
Ember is intended to bind to loopback by default; exposing it through a reverse
proxy with `--host`/`EMBER_HOST` is an operator decision and requires
appropriate authentication, request limits, and TLS at that boundary. The
supported ROCm container has direct GPU device access, host IPC, and an
unconfined seccomp profile. The XDNA overlay additionally exposes the NPU
device. Neither image is a sandbox for untrusted models or code.

Model files are external artifacts. Keep the default pinned revision and
startup SHA-256 verification enabled. `EMBER_VERIFY_EXISTING_SHA256=0` is only
appropriate when an external trusted, immutable artifact store already
enforces integrity; Ember still verifies every completed download before
promotion. When changing the pinned model in source, update its revision,
expected size, and digest together, and review its license and safety terms
separately from Ember's.

Ember sends no usage telemetry. Automatic model download contacts the
configured Hugging Face repository during startup.
