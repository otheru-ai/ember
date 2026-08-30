# Agent registry

Four agents work this repo. Names are addresses in WIRE `to=`/`from=`.

| name | role | reachable by | push latency |
|---|---|---|---|
| `claude` | review, independent verification, specs, target tracking | `.coord/msg/`; also `claude -p` relay to session `ember-30` | ~20s watcher; relay immediate |
| `codex` | implementation, GPU/CI runs, hardware evidence | `.coord/msg/`; also `codex queue --thread <id>`; also `turn/steer` over the app-server control socket | steer immediate; queue = turn boundary |
| `kimi` | GPU-free reproduction: make correctness failures testable on the host | `.coord/msg/` only | poll-based |
| `grok` | research: x.com, cutting-edge papers, upstream ggml/llama.cpp/ROCm developments | `.coord/msg/` only | **60-90s, minutes if mid-fetch** |

## Channel notes

- **`.coord/msg/` is the universal channel.** Every agent can read and write
  files. Use it by default for everything.
- **codex** additionally has an immediate push channel (`turn/steer` via
  `~/.codex/app-server-control/app-server-control.sock`, websocket JSON-RPC,
  requires `expectedTurnId`). Reserved for `!` urgency.
- **claude** additionally has an immediate push channel: a headless
  `claude -p --allowed-tools "SendMessage ListAgents"` relay that SendMessages
  session `ember-30`. Costs a whole Claude session per message; reserve for `!`.
- **grok has no inbound push channel and cannot get one** (confirmed by grok
  itself, build 1.0.13 `5e9a58528b76`):
  - `grok leader` is a shared backend multiplexer for config reload / MCP
    watches / relay. Verbs are only `list`/`info`/`kill`. Starting under
    `grok agent leader` does **not** give a codex-style mid-turn push.
  - `grok agent serve` is an ACP WebSocket server (default `127.0.0.1:2419`,
    auth `--secret`/`GROK_AGENT_SECRET`). Its lifecycle is
    `initialize` -> `session/new` -> `session/prompt`, which creates a **new**
    session. No documented method names an existing TUI session and injects
    mid-turn.
  - No schema generator exists (`--json-schema` is output constraint, not
    protocol). Grok extensions live under `x.ai/*`; none is a steer.
  - TUI `ui.follow_up_behavior = "steer"` is a human pressing Enter, not IPC.
  - **Practical latency:** grok lists `.coord/msg/` at task start, task finish,
    and ~60s intervals when not blocked in a long tool. No background watcher.
    Budget 60-90s; minutes during a long fetch. Do not assume sub-20s.

- **kimi has no inbound push channel.** Plain `kimi` TUI, no web server
  listening. `kimi acp` is ACP over **stdio** and creates a new session; it
  cannot attach to a running TUI. `kimi web` (default port 58627, bearer token)
  would start a *separate* server, not expose the running session. So kimi
  polls `.coord/msg/` like grok. Sessions live in
  `~/.kimi-code/sessions/wd_ember_*/`, indexed in `session_index.jsonl`.

## Talk to each other directly — do not route through claude

Any agent may address any other. `to=` accepts `claude`, `codex`, `grok`,
`kimi`, or `all`. Use it.

Specifically:

- **codex -> grok**: research questions. "Which shapes hit this path?", "Is
  this kernel safe at width N?", "Has anyone upstream measured this?" Send them
  straight to grok; do not wait for claude to relay. Add `to=all` or copy
  claude only if you want the answer verified against source.
- **grok -> codex**: findings that change what codex should run next, and
  answers to codex's questions. Do not hold research until claude relays it.
- **kimi -> codex**: questions about engine internals or hardware behaviour
  kimi cannot test on the host.
- **anything you want independently checked -> claude**, or `to=all` if the
  whole group should see it.

Claude verifying research against source has caught real errors and should
continue, but it must not be a *relay hop*. Use `to=all` so claude sees the
exchange without being in the path.

Historical note, so the pattern is not repeated: for the first day of this
project every message went through claude because the onboarding examples all
used `to=claude`. 40 claude->codex, 17 grok->claude, and **zero** grok<->codex.
That made claude a latency bottleneck on questions it had no special ability
to answer.

## Division of labour

- `codex` implements and measures. Does not count its own review as independent.
- `claude` reviews, verifies independently, and owns the review waterline.
- `grok` researches and cites. Does not commit code.
- `kimi` owns host-side reproducibility and test infrastructure. Does not
  dispatch workflows, touch the GPU, or change engine numerics.
- GPU time and production downtime are authorized (user decision, 2026-08-30).
