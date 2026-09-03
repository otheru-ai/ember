# Coding-agent clients

Ember exposes three compatible APIs on the same port:

- Anthropic Messages at `/v1/messages`
- OpenAI Responses at `/v1/responses`
- OpenAI Chat Completions at `/v1/chat/completions`

The examples below assume Ember is listening on its default API port,
`http://127.0.0.1:8080`, and advertising `deepseek-v4-flash`.

Release validation on 2026-08-08 completed real, isolated CLI turns with Claude
Code 2.1.224, Codex CLI 0.147.0, OpenCode 1.18.15, pi 0.84.1, and OMP 17.2.11.
Each client received a visible streamed answer from Ember's GPU-free backend.
The automated suite separately exercises tool-bearing request shapes and event
ordering without requiring those clients to be installed in CI. It also pins
the native DeepSeek Chat Completions request used by Reasonix 1.31.3 and
DeepSeek Harness `llm-deepseek` 0.1.1-rc.2, including
`thinking.type=enabled|disabled`.

## Claude Code

Claude Code uses the Anthropic Messages API. Launch it with:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_AUTH_TOKEN=ember-local \
ANTHROPIC_MODEL=deepseek-v4-flash \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude
```

`ember-local` is only a placeholder: Ember does not require authentication when
bound to loopback. Do not expose an unauthenticated Ember port to another host.
Claude Code's remote-only features, including Anthropic-hosted web search, are
not available through a local model endpoint.

## Codex CLI

Codex uses the Responses API. Add this to the user-level
`~/.codex/config.toml` (provider settings are not accepted in project config):

```toml
model = "deepseek-v4-flash"
model_provider = "ember"
web_search = "disabled"

[model_providers.ember]
name = "Ember"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
requires_openai_auth = false
```

Disabling web search is required. Codex's `web_search` is a hosted Responses
tool executed by OpenAI, not a function that a local inference server or the
Codex shell can execute. Codex's local shell, patch, and other client-executed
tools remain available.

## Reasonix

Install Reasonix and add Ember as its default provider:

```bash
npm install -g reasonix@1.31.3
mkdir -p ~/.reasonix
touch ~/.reasonix/.env
grep -qxF 'EMBER_API_KEY=sk-ember-local' ~/.reasonix/.env || \
  printf 'EMBER_API_KEY=sk-ember-local\n' >> ~/.reasonix/.env
```

```toml
# ~/.reasonix/config.toml
default_model = "ember"

[[providers]]
name = "ember"
kind = "openai"
base_url = "http://127.0.0.1:8080/v1"
model = "deepseek-v4-flash"
api_key_env = "EMBER_API_KEY"
context_window = 131072
reasoning_protocol = "deepseek"
supported_efforts = ["low", "high", "max"]
default_effort = "high"
```

`sk-ember-local` is a placeholder for the loopback-only Ember endpoint.
Reasonix's native DeepSeek mode sends the official `thinking` object together
with `reasoning_effort`; Ember accepts both fields directly.

## DeepSeek Harness

DeepSeek Harness's official adapter appends `/chat/completions` to its base URL,
so point its environment at Ember's `/v1` root:

```bash
DEEPSEEK_BASE_URL=http://127.0.0.1:8080/v1 \
DEEPSEEK_API_KEY=sk-ember-local \
npx @deepseek-ai/dsh web
```

Select `deepseek-v4-flash` in the harness. The placeholder key satisfies the
harness's client-side credential requirement; Ember does not authenticate it
on loopback. The adapter's streaming usage, `reasoning_content` replay, tools,
and `thinking.type` request fields are covered by Ember's compatibility suite.

## OpenCode

Add an Ember provider to `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "ember/deepseek-v4-flash",
  "provider": {
    "ember": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ember (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "ember-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash",
          "limit": {
            "context": 65536,
            "output": 16384
          }
        }
      }
    }
  }
}
```

Then start OpenCode with `opencode --model ember/deepseek-v4-flash` or select
the model with `/models`.

## pi

Add a custom provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ember": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "ember-local",
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 16384,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Run `pi --model ember/deepseek-v4-flash`.

## OMP (oh-my-pi)

Add a keyless provider to `~/.omp/agent/models.yml`:

```yaml
providers:
  ember:
    baseUrl: http://127.0.0.1:8080/v1
    api: openai-completions
    auth: none
    models:
      - id: deepseek-v4-flash
        name: DeepSeek V4 Flash
        reasoning: true
        input: [text]
        contextWindow: 65536
        maxTokens: 16384
        cost:
          input: 0
          output: 0
          cacheRead: 0
          cacheWrite: 0
```

Validate discovery with `omp models find ember`, then run
`omp --model ember/deepseek-v4-flash`.

## Compatibility scope

Ember supports streaming text, reasoning, function tools, parallel tool calls,
tool-result replay, stop sequences, and usage reporting across these adapters.
Native Chat Completions also accepts `image_url` content parts and routes them
through the opt-in vision seam. The Responses adapter rejects
`input_image`, and the Anthropic Messages adapter currently accepts only
`type: text` content blocks. Audio, computer-use, and provider-hosted tools are
not implemented. Vision remains an opt-in deployment boundary, not a certified
multimodal release; see `docs/operations.md`. Client releases can change their wire behavior, so the
`client_compatibility_server` test keeps representative Anthropic Messages,
Responses, and Chat Completions requests in the release test suite, including
the native DeepSeek request shared by Reasonix and DeepSeek Harness.
