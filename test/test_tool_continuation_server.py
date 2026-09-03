#!/usr/bin/env python3
"""End-to-end cover for the bound tool-result continuation path.

A continuation request carries only `role: "tool"` messages and no history at
all. The server is then expected to answer from state it stored when it emitted
the call, rather than re-rendering a conversation it was never sent. That is the
exact-token tool replay invariant, and it has two halves worth pinning:

  * a continuation whose call id the server never issued must be refused with a
    409 telling the client to replay the full history -- never answered from a
    guess, and never a 500;
  * a continuation whose call id the server did issue must be answered, with the
    stored prompt reused rather than rebuilt.

Both run against the GPU-free stub backend.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


PIPE = "｜"
OPEN = f"<{PIPE}DSML{PIPE}"
CLOSE = f"</{PIPE}DSML{PIPE}"

g_pass = 0
g_fail = 0


def check(condition: bool, message: str) -> None:
    global g_pass, g_fail
    if condition:
        g_pass += 1
        print(f"  [PASS] {message}")
    else:
        g_fail += 1
        print(f"  [FAIL] {message}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(url: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read()
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw.decode()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw.decode()


def wait_ready(base: str, proc: subprocess.Popen) -> None:
    # Generous on purpose: the same binary under ASan takes several seconds to
    # reach /health, and a ready window tuned to the plain build turns the
    # sanitizer job into a flaky failure that says nothing about the code.
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(
                f"stub server exited before ready (status {proc.returncode}): "
                f"{proc.stderr.read()}"
            )
        try:
            status, _ = request(base + "/health")
            if status == 200:
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise RuntimeError("stub server did not become ready")


def dsml_call(name: str, params: list[tuple[str, str]]) -> str:
    body = "".join(
        f'{OPEN}parameter name="{key}" string="true">{value}{CLOSE}parameter>'
        for key, value in params
    )
    return (
        f"{OPEN}tool_calls>"
        f'{OPEN}invoke name="{name}">{body}{CLOSE}invoke>'
        f"{CLOSE}tool_calls>"
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]


def chat_request(messages: list[dict]) -> dict:
    return {
        "model": "stub",
        "stream": False,
        "max_tokens": 4096,
        "reasoning_effort": "none",
        "messages": messages,
        "tools": TOOLS,
    }


class Server:
    """One stub server, kept alive across requests so state can carry over."""

    def __init__(self, binary: str, env_extra: dict):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = os.environ.copy()
        env.update(env_extra)
        self.proc = subprocess.Popen(
            [binary, "-m", "stub", "--port", str(self.port),
             "--ds4-prefill", "exact"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        wait_ready(self.base, self.proc)

    def chat(self, messages: list[dict]):
        return request(self.base + "/v1/chat/completions",
                       chat_request(messages))

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


def test_unknown_call_id_is_refused_not_guessed(binary: str) -> None:
    print("continuation with a call id the server never issued")
    server = Server(binary, {"EMBER_STUB_REPLY": "done"})
    try:
        status, body = server.chat([
            {"role": "tool", "tool_call_id": "call_never_issued",
             "content": "the tool result"},
        ])
        check(status == 409, f"refused with 409 (got {status})")
        detail = json.dumps(body)
        check("continuation_state_unavailable" in detail,
              "names continuation_state_unavailable so a client can branch on it")
        check("replaying the full message history" in detail,
              "tells the client what to do instead of failing opaquely")
    finally:
        server.close()


def test_bound_continuation_answers_from_stored_state(binary: str) -> None:
    print("continuation with the call id the server did issue")
    server = Server(binary, {
        "EMBER_STUB_REPLY": dsml_call("lookup", [("key", "alpha")]),
        # The reply used once the tool result comes back. Without it the
        # continuation would return the tool call again and the test could pass
        # without the stored prompt ever being reused.
        "EMBER_STUB_RECOVERY_REPLY": "the answer is alpha",
    })
    try:
        status, body = server.chat([
            {"role": "user", "content": "look up alpha"},
        ])
        check(status == 200, f"the first turn succeeds (got {status})")
        calls = (body.get("choices", [{}])[0]
                     .get("message", {})
                     .get("tool_calls") or [])
        check(len(calls) == 1, f"the first turn emits one tool call (got {len(calls)})")
        if not calls:
            return
        call_id = calls[0]["id"]
        check(bool(call_id), "the emitted call carries an id")

        # Only the tool result. No user turn, no assistant turn: if the server
        # were re-rendering a conversation rather than reusing stored state,
        # there would be nothing here to render.
        status, body = server.chat([
            {"role": "tool", "tool_call_id": call_id, "content": "alpha=42"},
        ])
        check(status == 200,
              f"the bound continuation is answered, not refused (got {status})")
        choice = body.get("choices", [{}])[0]
        again = choice.get("message", {}).get("tool_calls") or []
        # This stub answers every turn with the same DSML block, so the
        # continuation produces a second call rather than prose. What matters is
        # that it generated at all, and that the new call got its own id: an id
        # equal to the first would mean the stored response was handed back
        # verbatim instead of the stored prompt being decoded again.
        check(len(again) == 1,
              f"the continuation generates a fresh turn (got {len(again)} calls)")
        if again:
            check(again[0]["id"] != call_id,
                  "the new call gets its own id, so this is a real generation")
        check(choice.get("finish_reason") == "tool_calls",
              "the continuation reports an ordinary finish reason")
    finally:
        server.close()


def test_full_history_replay_splices_the_remembered_block(binary: str) -> None:
    """The other half of the 409's advice: replaying the whole history works.

    The renderer does not re-serialise a remembered assistant tool call into the
    prompt text; it emits a splice sentinel that the encoder resolves back to the
    exact tokens that were sampled. Getting that wrong changes prompt token
    identity between the turn that made the call and the turn that answers it,
    which silently invalidates the KV prefix.
    """
    print("full history replay resolves the remembered tool block")
    server = Server(binary, {
        "EMBER_STUB_REPLY": dsml_call("lookup", [("key", "beta")]),
    })
    try:
        status, body = server.chat([{"role": "user", "content": "look up beta"}])
        check(status == 200, f"the first turn succeeds (got {status})")
        calls = (body.get("choices", [{}])[0]
                     .get("message", {}).get("tool_calls") or [])
        if not calls:
            check(False, "the first turn emits a tool call")
            return
        call = calls[0]

        # The full history a well-behaved client replays: the user turn, the
        # assistant turn carrying the call verbatim, then the result.
        status, body = server.chat([
            {"role": "user", "content": "look up beta"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "beta=7"},
        ])
        check(status == 200,
              f"the replayed history is accepted (got {status})")
        again = (body.get("choices", [{}])[0]
                     .get("message", {}).get("tool_calls") or [])
        check(len(again) == 1,
              f"the replayed turn generates (got {len(again)} calls)")
        if again:
            check(again[0]["id"] != call["id"],
                  "the replayed turn samples afresh rather than echoing the call")
    finally:
        server.close()


def test_unknown_id_in_replayed_history_still_renders(binary: str) -> None:
    """A sentinel the server cannot resolve must not take the request down.

    A client can replay a history from a previous process, or edit a call id by
    hand. The block is then unknown, there is nothing to splice, and the correct
    outcome is an ordinary render -- not a 500 and not a silent truncation.
    """
    print("replayed history naming a call this process never issued")
    server = Server(binary, {"EMBER_STUB_REPLY": "plain answer"})
    try:
        status, _ = server.chat([
            {"role": "user", "content": "look up gamma"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_from_a_previous_process",
                "type": "function",
                "function": {"name": "lookup",
                             "arguments": "{\"key\":\"gamma\"}"},
            }]},
            {"role": "tool", "tool_call_id": "call_from_a_previous_process",
             "content": "gamma=1"},
        ])
        check(status == 200,
              f"an unresolvable block renders normally (got {status})")
    finally:
        server.close()


def test_mixed_history_is_not_a_continuation(binary: str) -> None:
    print("a request carrying more than tool results is ordinary history")
    server = Server(binary, {"EMBER_STUB_REPLY": "ordinary answer"})
    try:
        # One non-tool message disqualifies the whole request, so this must take
        # the ordinary render path and succeed rather than 409 on missing state.
        status, _ = server.chat([
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "call_never_issued",
             "content": "stray result"},
        ])
        check(status == 200,
              f"mixed history renders normally instead of 409 (got {status})")
    finally:
        server.close()


def test_duplicate_results_for_one_call(binary: str) -> None:
    print("two tool results carrying the same call id")
    server = Server(binary, {"EMBER_STUB_REPLY": "done"})
    try:
        # The ids collapse to one binding, so this is still refused for the same
        # single unknown id -- not treated as two, and not a crash.
        status, body = server.chat([
            {"role": "tool", "tool_call_id": "call_dup", "content": "first"},
            {"role": "tool", "tool_call_id": "call_dup", "content": "second"},
        ])
        check(status == 409, f"still one unknown binding, refused (got {status})")
        check("continuation_state_unavailable" in json.dumps(body),
              "the duplicate collapses rather than producing a different error")
    finally:
        server.close()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: test_tool_continuation_server.py <ember-server>",
              file=sys.stderr)
        return 2
    binary = sys.argv[1]
    print("tool continuation server tests\n")
    test_unknown_call_id_is_refused_not_guessed(binary)
    test_bound_continuation_answers_from_stored_state(binary)
    test_full_history_replay_splices_the_remembered_block(binary)
    test_unknown_id_in_replayed_history_still_renders(binary)
    test_mixed_history_is_not_a_continuation(binary)
    test_duplicate_results_for_one_call(binary)
    print(f"\n{g_pass} passed, {g_fail} failed")
    return 1 if g_fail else 0


if __name__ == "__main__":
    sys.exit(main())
