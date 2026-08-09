# Security policy

Please do not disclose vulnerabilities in an issue. Use the repository
host's private security-advisory feature. If that feature is unavailable, email
`otheru@otheru.ai` with `Ember security` in the subject. Include the affected
revision, reproduction steps, impact, and any proposed mitigation. Do not attach
model weights, private prompts, credentials, or operational data.

Only the latest release and the current `main` branch receive security fixes.
Ember is intended to bind to loopback by default; exposing it through a reverse
proxy is an operator decision and requires appropriate authentication, request
limits, and TLS at that boundary. The supported ROCm container has direct GPU
device access, host IPC, and an unconfined seccomp profile, so it is not a
sandbox for untrusted models or code.

Model files are external artifacts. Keep the default pinned revision and
SHA-256 verification enabled. When selecting another model, update its
revision, expected size, and digest together, and review its license and safety
terms separately from Ember's.

Ember sends no usage telemetry. Automatic model download contacts the
configured Hugging Face repository during startup.
