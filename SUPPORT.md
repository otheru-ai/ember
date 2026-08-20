# Support

Use repository issues for reproducible bugs and focused feature requests. Include
the Ember revision, OS/kernel, ROCm version, GPU architecture, exact command,
relevant logs, and whether the GPU-free test suite passes.

For the optional XDNA path, also include the `amdxdna` driver, firmware and XRT
versions, IOMMU state, `/dev/accel/accel0` permissions, provider environment,
and whether fallback was allowed. State whether the synthetic provider gate or
trained two-session differential failed; NPU activity alone is not evidence of
correct execution.

Use discussions for setup help and design questions. Security vulnerabilities
must follow `SECURITY.md` and must not be filed in an issue.

The maintainers cannot redistribute model weights or provide support for model
licenses, host GPU/NPU drivers or firmware, unsupported hardware, or third-party
reverse proxies.
