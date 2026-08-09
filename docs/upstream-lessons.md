# Upstream release lessons

Ember is deliberately narrower than general inference platforms, but narrow
scope is not an excuse for a fragile release. This review records the
deployment patterns considered for the initial release and the decisions made
from them. Links point to upstream project documentation, not secondary setup
guides.

## Comparative review

| Project | Useful release pattern | Ember decision |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Multiple distribution forms, server-specific images, direct Hugging Face model acquisition, checksum-bearing release artifacts, and an explicit untrusted-model threat model. | Keep one target-specific image and direct model acquisition. Pin and hash the single supported model. Preserve the explicit model/server security boundary. Published binaries and images remain a release-pipeline task. |
| [DwarfStar/ds4](https://github.com/antirez/ds4) | Honest model/hardware scope, a model download helper, first-class OpenAI/Anthropic agent examples, differential/debug tooling, and test vectors based on real model behavior. | Match the narrow-scope honesty, agent-client recipes, validator, and model-specific QA. Do not copy ds4 features that conflict with documented Ember fidelity decisions. |
| [vLLM](https://docs.vllm.ai/en/latest/deployment/docker/) | Stable and nightly ROCm images are visibly distinct; GPU/cache mounts are documented; non-root operation is supported; production metrics cover queueing, cache, throughput, TTFT, and latency. | Add OCI version/revision labels, persistent model/cache mounts, health checks, and an explicit observability gap. A non-root target needs target-host permission testing before it can be claimed. Prometheus semantics require a designed, versioned contract rather than wrapping `/status` casually. |
| [SGLang](https://docs.sglang.io/docs/get-started/install) | Immutable image tags are recommended over `latest`, runtime images exclude development tools, and hardware-specific installation paths are explicit. | Document immutable release metadata, keep hardware specificity prominent, and separate the full ROCm development stage from a minimal runtime carrying the exact linked-library closure. |
| [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/reference/launcher) | Model revisions are first-class, request/concurrency limits are explicit, cache locations are mountable, metrics are documented, and usage telemetry is disclosed with an opt-out. | Pin the Hugging Face commit and digest, retain Ember's bounded queue/backpressure, persist cache paths, and state clearly that Ember sends no telemetry. Metrics remain deferred as above. |
| [Ollama](https://github.com/ollama/ollama) | Model pull/run is the product path, Docker is official, the API has model/version discovery, and coding-agent integrations are treated as first-class onboarding. | Keep automatic resumable acquisition and minimize commands; provide health/model/status smoke tests and tested client configurations. Do not introduce a general model registry for a one-model engine. |
| [LocalAI](https://localai.io/docs/basics/getting_started/) | Quickstart separates hardware images and warns that remotely exposed APIs need authentication; OpenAI and Anthropic compatibility is documented directly. | Retain loopback-only defaults, state that Ember has no built-in auth, require an authenticating proxy for remote use, and test both protocol families plus coding-agent clients. |

## Integrated for the initial release

- A one-command Compose path with persistent model and KV-cache directories.
- Host and in-container preflight checks before a very large download.
- Immutable model revision, expected size, and SHA-256 verification.
- Resumable staging followed by atomic promotion of a verified model.
- Compose health status and a separate read-only/end-to-end smoke test.
- OCI image version, revision, license, title, and vendor metadata.
- Separate `dev` and `release` image targets; only the former carries the full
  ROCm toolchain, source, and build tree.
- A dedicated operations guide with hardware, update, rollback, security, and
  troubleshooting contracts.
- Explicit disclosure of the loopback/no-auth boundary and absence of telemetry.
- Compatibility recipes and regression tests for common coding-agent clients.
- GPU-free regression tests for the release scripts themselves.

## Deliberately deferred

- **Published multi-platform binaries:** the real engine is a `gfx1151` ROCm
  target and must be validated on that hardware. A generic binary would imply
  support Ember does not have.
- **Non-root-by-default container:** model/cache ownership and host GPU group
  mappings vary. It will be added only with a tested UID/GID contract.
- **Built-in API keys/TLS:** loopback is the supported trust boundary. Remote
  service authentication belongs at a hardened proxy until Ember has a reviewed
  credential lifecycle and constant-time authorization path.
- **Prometheus/OpenTelemetry:** metrics names, counter lifetimes, cardinality,
  privacy, and compatibility need an intentional API. `/status` remains the
  initial diagnostic surface.
- **Kubernetes charts and autoscaling:** the target is one large model on one
  Strix Halo host. Shipping orchestration templates before the single-node
  lifecycle is proven would create an unsupported deployment promise.
- **A general model gallery:** Ember's tuned kernels and tokenizer are coupled
  to the published DeepSeek-V4-Flash artifact. Accepting arbitrary GGUF files
  would weaken both safety and supportability.

This document should be revisited for every release that changes supported
hardware, model artifacts, container privileges, API exposure, or operational
telemetry.
