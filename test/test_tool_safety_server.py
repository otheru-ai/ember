#!/usr/bin/env python3
"""End-to-end proof that unsafe generated tool calls never reach a harness."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


PIPE = "\uff5c"
OPEN = f"<{PIPE}DSML{PIPE}"
CLOSE = f"</{PIPE}DSML{PIPE}"


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
        with urllib.request.urlopen(req, timeout=5) as response:
            raw = response.read()
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw.decode()
            return response.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def wait_ready(base: str, proc: subprocess.Popen) -> None:
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError(
                f"stub server exited before ready "
                f"(status {proc.returncode}): {proc.stderr.read()}"
            )
        try:
            status, _ = request(base + "/health")
            if status == 200:
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.03)
    raise RuntimeError("stub server did not become ready")


def tool_request(*, stream: bool = False) -> dict:
    return {
        "model": "stub",
        "stream": stream,
        "max_tokens": 4096,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": "write the requested file"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ],
    }


def responses_tool_request() -> dict:
    return {
        "model": "stub",
        "stream": True,
        "max_output_tokens": 4096,
        "reasoning": {"effort": "none"},
        "input": "write the requested file",
        "tools": [
            {
                "type": "function",
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            }
        ],
    }


def dsml_write(*, path: str | None, content: str) -> str:
    params = []
    if path is not None:
        params.append(
            f'{OPEN}parameter name="path" string="true">{path}'
            f"{CLOSE}parameter>"
        )
    params.append(
        f'{OPEN}parameter name="content" string="true">{content}'
        f"{CLOSE}parameter>"
    )
    return (
        f"{OPEN}tool_calls>\n"
        f'{OPEN}invoke name="write_file">\n'
        + "\n".join(params)
        + f"\n{CLOSE}invoke>\n{CLOSE}tool_calls>"
    )


def dsml_call(name: str, parameters: list[tuple[str, bool, str]]) -> str:
    encoded = []
    for key, is_string, value in parameters:
        encoded.append(
            f'{OPEN}parameter name="{key}" '
            f'string="{"true" if is_string else "false"}">{value}'
            f"{CLOSE}parameter>"
        )
    return (
        f"{OPEN}tool_calls>"
        f'{OPEN}invoke name="{name}">'
        + "".join(encoded)
        + f"{CLOSE}invoke>{CLOSE}tool_calls>"
    )


def run_case(
    server: str,
    initial: str,
    recovery: str,
    *,
    stream: bool = False,
    endpoint: str = "/v1/chat/completions",
    payload: dict | None = None,
    termination_reason: str | None = None,
    think_tool_reply: str | None = None,
    recovery_error: str | None = None,
    extra_env: dict | None = None,
):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = initial
    env["EMBER_STUB_RECOVERY_REPLY"] = recovery
    if termination_reason is not None:
        env["EMBER_STUB_TERMINATION_REASON"] = termination_reason
    if think_tool_reply is not None:
        env["EMBER_STUB_THINK_TOOL_REPLY"] = think_tool_reply
    if recovery_error is not None:
        env["EMBER_STUB_RECOVERY_ERROR"] = recovery_error
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        [
            server,
            "-m",
            "stub",
            "--port",
            str(port),
            "--ds4-prefill",
            "exact",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base, proc)
        return request(
            base + endpoint,
            payload if payload is not None else tool_request(stream=stream),
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def run_tool_loop_observability_case(server: str, reply: str) -> None:
    """The report is additive: the tool call and standard finish enum survive."""
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = reply
    # Explicit --tool-loop-report 3: this case exercises the REPORT MECHANISM
    # with a 4-round history, not the shipped default of 8.
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port), "--ds4-prefill", "exact",
         "--tool-loop-report", "3"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base, proc)
        messages = [{"role": "user", "content": "write it"}]
        for i in range(1, 5):
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": (
                                        '{"path":"/tmp/safe.py",'
                                        '"content":"same"}'
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{i}",
                        "content": "ok",
                    },
                ]
            )
        payload = tool_request(stream=False)
        payload["messages"] = messages
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls", body
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "write_file", body
        assert choice["ember_tool_loop"] == {
            "rounds": 4,
            "tool": "write_file",
            "identical_results": True,
        }, body

        code, status = request(base + "/status")
        assert code == 200, status
        assert status["tool_loop"]["report_after_repeats"] == 3, status
        assert status["tool_loop"]["last"]["rounds"] == 4, status
        assert status["tool_loop"]["last"]["tool"] == "write_file", status

        payload["stream"] = True
        code, stream = request(base + "/v1/chat/completions", payload)
        assert code == 200, stream
        assert '"finish_reason":"tool_calls","ember_tool_loop":' in stream, stream
        assert '"rounds":4,"tool":"write_file","identical_results":true' in stream, stream
        assert '"finish_reason":"tool_loop"' not in stream, stream
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert (
        '[ember] tool loop: 4 identical call+result rounds for "write_file"'
        in log
    ), log


def run_nonprogress_case(server: str) -> None:
    """A turn that finishes clean but emits nothing at all must be reported.

    A model can produce reasoning tokens and then no visible text or tool call.
    Backend flags alone are insufficient; the condition must appear in logs and
    status telemetry.
    """
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    # Whitespace-only: the model DOES decode tokens but delivers no visible
    # byte and no tool call -- the real deployment shape (~700 tokens of
    # reasoning, then silence). An earlier version of this test used "" which
    # decodes ZERO tokens; that is a different thing entirely (a request that
    # asked for no output), and building the test on it is what let the
    # too-loose n_generated==0 case ship. sse.h defines sent_visible_content
    # as "a non-whitespace content byte reached", so whitespace stays invisible.
    env["EMBER_STUB_REPLY"] = "   \n  \t "
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base, proc)
        code, status = request(base + "/status")
        assert code == 200, status
        assert status["nonprogress"]["count"] == 0, status
        assert status["nonprogress"]["last"] is None, status

        # buffered path
        payload = tool_request(stream=False)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        assert body["choices"][0]["message"]["content"].strip() == "", body
        code, status = request(base + "/status")
        assert status["nonprogress"]["count"] == 1, status
        assert status["nonprogress"]["last"] is not None, status

        # streaming path counts independently
        payload["stream"] = True
        code, _ = request(base + "/v1/chat/completions", payload)
        assert code == 200
        code, status = request(base + "/status")
        assert status["nonprogress"]["count"] == 2, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "[ember] non-progress turn:" in log, log


def run_zero_token_not_counted_case(server: str) -> None:
    """A request that decoded NOTHING is not the model refusing to act.

    A generation that never decoded is not a non-progress turn; it is a
    separate no-output condition caused by a zero budget, immediate stop, or
    abandonment.
    """
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = ""          # decodes ZERO tokens
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base, proc)
        payload = tool_request(stream=False)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        payload["stream"] = True
        request(base + "/v1/chat/completions", payload)
        code, status = request(base + "/status")
        assert status["nonprogress"]["count"] == 0, status
        assert status["nonprogress"]["last"] is None, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "non-progress turn" not in log, log


def run_length_truncated_not_counted_case(server: str) -> None:
    """A turn cut off by the token cap is a budget outcome, not a refusal.

    A capped request may produce reasoning tokens without visible output or a
    tool call. Only a voluntary stop should count as non-progress.
    """
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "   \n  \t "     # invisible, but decodes tokens
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base, proc)
        payload = tool_request(stream=False)
        payload["max_tokens"] = 2               # force finish_reason "length"
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        assert body["choices"][0]["finish_reason"] == "length", body
        code, status = request(base + "/status")
        assert status["nonprogress"]["count"] == 0, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "non-progress turn" not in log, log


def _loop_messages(n: int, name: str = "write_file") -> list:
    """n identical assistant tool calls, each with a DIFFERENT result.

    Differing results are invisible to a strict call-and-result signal, so the
    call-signature diagnostic must cover them separately.
    """
    msgs = [{"role": "user", "content": "do the thing"}]
    for i in range(1, n + 1):
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": f"c{i}", "type": "function",
                "function": {"name": name,
                             "arguments": '{"path":"/tmp/x","content":"same"}'},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": f"result number {i}"})   # differs each round
    return msgs


def run_ordinary_tool_call_is_not_a_leak_case(server: str) -> None:
    """An ordinary STREAMING tool call must not count as a markup leak (#24).

    The counter used to scan the caller's raw accumulator, which still holds
    the tool block that ember_text_safe_limit() correctly withheld and the
    parser then consumed. That incorrectly counted ordinary tool calls as leaks.

    The existing prose case asserts count==0 for a BUFFERED, tool-free reply,
    which never exercised the defect. This one does: streaming, with tools,
    emitting a real DSML call.
    """
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = dsml_write(path="/tmp/safe.py", content="print(1)\n")
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        wait_ready(base, proc)
        code, body = request(base + "/v1/chat/completions", tool_request(stream=True))
        assert code == 200, body
        assert "tool_calls" in body, body       # the call really was emitted
        code, status = request(base + "/status")
        assert status["tool_markup_leak"]["count"] == 0, status

        # and again buffered, so neither path regresses
        code, body = request(base + "/v1/chat/completions", tool_request(stream=False))
        assert code == 200, body
        assert body["choices"][0]["message"].get("tool_calls"), body
        code, status = request(base + "/status")
        assert status["tool_markup_leak"]["count"] == 0, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "tool markup leaked" not in log, log


def run_loop_report_defaults_case(server: str) -> None:
    """Both loop thresholds default to 8.

    Lower thresholds were noisy for loops that resolved when the model changed
    strategy. Pin the default here because this test passes neither flag.
    """
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        wait_ready(base, proc)
        code, status = request(base + "/status")
        assert status["tool_loop"]["report_after_repeats"] == 8, status
        assert status["no_progress"]["report_after"] == 8, status

        # A six-round wall -- the shape that self-resolved in deployment -- is
        # now BELOW the threshold and must stay silent on both signals.
        payload = tool_request(stream=False)
        payload["messages"] = _wall_messages(6)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        code, status = request(base + "/status")
        assert status["no_progress"]["count"] == 0, status
        assert status["tool_loop"]["last"] is None, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _wall_messages(n: int, name: str = "terminal") -> list:
    """n tool rounds with a DIFFERENT call each time and one invariant answer.

    A captured regression showed: the model varied its
    terminal command every round and the harness returned a byte-identical
    "Background review denied non-whitelisted tool" refusal twelve times. Both
    call-keyed signals are near-blind to this (measured on the regression fixture:
    rounds=0, calls=2) because they ask whether the CALL repeated. The lease
    asks whether anything came back that had not come back before.
    """
    wall = ('{"error": "Background review denied non-whitelisted tool: '
            f'{name}. Only memory/skill tools are allowed."}}')
    msgs = [{"role": "user", "content": "do the thing"}]
    for i in range(1, n + 1):
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": f"w{i}", "type": "function",
                "function": {"name": name,
                             "arguments": '{"command":"attempt %d"}' % i},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": f"w{i}", "content": wall})
    return msgs


def run_progress_lease_case(server: str) -> None:
    """Trailing tool rounds that return nothing new must be reported.

    This is a sibling of tool_loop rather than part of it because result-keyed
    and call-keyed signals detect different failure modes.
    """
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "plain answer"
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port), "--no-progress-report", "3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        wait_ready(base, proc)
        code, status = request(base + "/status")
        assert status["no_progress"]["report_after"] == 3, status
        assert status["no_progress"]["count"] == 0, status
        assert status["no_progress"]["last"] is None, status

        # Four rounds == a lease of 3 (the first introduced the refusal), which
        # is the threshold and not above it. Must stay silent.
        payload = tool_request(stream=False)
        payload["messages"] = _wall_messages(4)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        code, status = request(base + "/status")
        assert status["no_progress"]["count"] == 0, status

        # Five rounds -> lease 4 -> above the threshold.
        payload["messages"] = _wall_messages(5)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        code, status = request(base + "/status")
        assert status["no_progress"]["count"] == 1, status
        assert status["no_progress"]["last"]["rounds"] == 4, status
        assert status["no_progress"]["last"]["tool"] == "terminal", status

        # The signal must not change generation: the turn is served normally.
        assert body["choices"][0]["finish_reason"] in ("stop", "tool_calls"), body

        # Genuine progress -- every round brings back something new -- must not
        # fire however long the run is. This is _loop_messages: an IDENTICAL
        # call each round with a DIFFERENT result, i.e. exactly the history the
        # call-keyed signal fires on and the lease must not.
        payload["messages"] = _loop_messages(9)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        code, status = request(base + "/status")
        assert status["no_progress"]["count"] == 1, status   # unchanged
        assert status["tool_loop"]["last"]["rounds"] > 3, status  # the other did
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "[ember] no progress: 4 tool rounds returned nothing new" in log, log


def run_progress_lease_off_case(server: str) -> None:
    """--no-progress-report 0 disables the signal entirely."""
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "plain answer"
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port), "--no-progress-report", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        wait_ready(base, proc)
        payload = tool_request(stream=False)
        payload["messages"] = _wall_messages(12)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        code, status = request(base + "/status")
        assert status["no_progress"]["report_after"] == 0, status
        assert status["no_progress"]["count"] == 0, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "[ember] no progress:" not in log, log


def run_auto_answer_case(server: str) -> None:
    """--auto-answer-after-loop suppresses tools once the run exceeds N."""
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = dsml_write(path="/tmp/safe.py", content="x")
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port),
         "--auto-answer-after-loop", "3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        wait_ready(base, proc)
        code, status = request(base + "/status")
        assert status["auto_answer"]["armed_after"] == 3, status
        assert status["auto_answer"]["count"] == 0, status

        # 3 identical calls == the threshold, NOT above it -> must not fire
        payload = tool_request(stream=False)
        payload["messages"] = _loop_messages(3)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        assert body["choices"][0]["message"].get("tool_calls"), body
        code, status = request(base + "/status")
        assert status["auto_answer"]["count"] == 0, status

        # 4 identical calls -> above the threshold -> tools suppressed.
        # Assert on what EMBER did, not on what the stub emitted: the stub
        # replays its canned DSML reply regardless of suppression, so ember
        # rightly rejects "a tool call was generated when no tools were
        # available". A real model cannot emit a call for tools it was never
        # shown -- has_tools gates both the prompt render and the grammar.
        # What matters here is that no tool call reaches the CLIENT.
        payload["messages"] = _loop_messages(4)
        code, body = request(base + "/v1/chat/completions", payload)
        if "choices" in body:
            assert not body["choices"][0]["message"].get("tool_calls"), body
        code, status = request(base + "/status")
        assert status["auto_answer"]["count"] == 1, status
        assert status["auto_answer"]["last"]["tool"] == "write_file", status

        # an explicit client demand for a tool must NEVER be overridden
        payload["messages"] = _loop_messages(6)
        payload["tool_choice"] = "required"
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        code, status = request(base + "/status")
        assert status["auto_answer"]["count"] == 1, status   # unchanged
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "[ember] auto-answer:" in log, log


def run_auto_answer_instruction_case(server: str) -> None:
    """The suppressed turn must TELL the model why its tools vanished.

    Removing tools silently can make a model improvise ASCII tool markup into
    visible output (<?DSML?tool_calls>, U+003F not U+FF5C). The
    instruction is what makes suppression comprehensible rather than merely
    obstructive.
    """
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "plain answer"
    # The stub swaps its reply only when the instruction is present in the
    # rendered prompt, so seeing this text back PROVES the instruction was
    # injected -- not merely that the tools were removed.
    env["EMBER_STUB_AUTOANSWER_REPLY"] = "INSTRUCTION-REACHED-THE-MODEL"
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port),
         "--auto-answer-after-loop", "3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        wait_ready(base, proc)
        payload = tool_request(stream=False)
        payload["messages"] = _loop_messages(4)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        assert body["choices"][0]["message"]["content"] == \
            "INSTRUCTION-REACHED-THE-MODEL", body
        code, status = request(base + "/status")
        assert status["auto_answer"]["count"] == 1, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "[ember] auto-answer:" in log, log
    assert "instruction not appended" not in log, log


def run_clean_output_not_flagged_case(server: str) -> None:
    """Prose that merely MENTIONS DSML must not be flagged.

    Only the NEGATIVE case is covered. The positive one -- markup actually
    reaching the client as text -- could not be faithfully reproduced with the
    stub: with tools advertised ember rejects it as invalid_tool_call, with no
    tools on the buffered path likewise, and on the streaming path the stub's
    block is dropped entirely rather than delivered. Deployment (a regression fixture,
    streaming, tools suppressed mid-request by auto-answer) DID deliver it, so
    that path differs from anything the stub can construct. Shipping a
    positive test built on an unfaithful redeployment is the same mistake that
    produced the v1 failure, so it is left uncovered and stated instead.
    """
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = (
        "The DSML markers are special tokens. A DSML tool call is parsed, "
        "never shown. See docs for DSML details."
    )
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        wait_ready(base, proc)
        payload = {
            "model": "stub", "stream": False, "max_tokens": 4096,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "explain DSML"}],
        }
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        code, status = request(base + "/status")
        assert status["tool_markup_leak"]["count"] == 0, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "tool markup leaked" not in log, log


def run_auto_answer_off_by_default_case(server: str) -> None:
    """Without the flag it must never fire, however long the loop runs."""
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = dsml_write(path="/tmp/safe.py", content="x")
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],      # no flag
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        wait_ready(base, proc)
        payload = tool_request(stream=False)
        payload["messages"] = _loop_messages(9)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        assert body["choices"][0]["message"].get("tool_calls"), body
        code, status = request(base + "/status")
        assert status["auto_answer"]["armed_after"] == 0, status
        assert status["auto_answer"]["count"] == 0, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "auto-answer" not in log, log


def run_degenerate_with_output_case(server: str) -> None:
    """A degenerate turn that DID produce output must still be counted.

    A compaction summarizer can degenerate while still emitting visible text,
    so the narrower non-progress signal correctly stays silent while the wider
    degenerate counter fires.
    """
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "a real answer the model did emit"
    env["EMBER_STUB_TERMINATION_REASON"] = "prompt_echo_detected"
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base, proc)
        code, status = request(base + "/status")
        assert status["degenerate"]["count"] == 0, status

        # STREAMING path only. On the buffered path a watchdog-stopped turn
        # returns a typed model_output_error by design, so it never reaches the
        # normal response path -- and it is already visible as an error. The
        # deployment case (the compaction summarizer) was streaming, where the
        # turn finishes normally and the degeneracy would otherwise be silent.
        payload = tool_request(stream=True)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        assert '"content"' in str(body), body          # output WAS produced
        code, status = request(base + "/status")
        assert status["degenerate"]["count"] == 1, status
        last = status["degenerate"]["last"]
        assert last["produced_output"] is True, status
        assert last["reason"] == "prompt_echo_detected", status
        # and it must NOT be counted as non-progress -- it emitted.
        assert status["nonprogress"]["count"] == 0, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "[ember] degenerate turn WITH output:" in log, log


def run_progress_not_flagged_case(server: str) -> None:
    """The negative half: a turn that DOES emit must never be counted."""
    port = free_port()
    base = f"http://{'127.0.0.1'}:{port}"
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "here is the answer"
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base, proc)
        payload = tool_request(stream=False)
        code, body = request(base + "/v1/chat/completions", payload)
        assert code == 200, body
        assert body["choices"][0]["message"]["content"], body
        payload["stream"] = True
        request(base + "/v1/chat/completions", payload)
        code, status = request(base + "/status")
        assert status["nonprogress"]["count"] == 0, status
        assert status["nonprogress"]["last"] is None, status
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log = proc.stderr.read()
    assert "non-progress turn" not in log, log


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: test_tool_safety_server.py EMBER_SERVER", file=sys.stderr)
        return 2
    server = sys.argv[1]
    valid = dsml_write(path="/tmp/safe.py", content="print('safe')\n")
    run_tool_loop_observability_case(server, valid)
    run_ordinary_tool_call_is_not_a_leak_case(server)
    run_loop_report_defaults_case(server)
    run_progress_lease_case(server)
    run_progress_lease_off_case(server)
    run_auto_answer_case(server)
    run_auto_answer_instruction_case(server)
    run_clean_output_not_flagged_case(server)
    run_auto_answer_off_by_default_case(server)
    run_nonprogress_case(server)
    run_zero_token_not_counted_case(server)
    run_length_truncated_not_counted_case(server)
    run_degenerate_with_output_case(server)
    run_progress_not_flagged_case(server)

    # Reproduce the deployment failure shape: ASCII-degraded DSML embedded in a
    # write_file string. The outer call must be rejected, and only the clean
    # replacement turn may be exposed.
    nested = dsml_write(
        path="/tmp/bad.py",
        content=(
            "partial source\n"
            "<?DSML?tool_calls><?DSML?invoke name=\"terminal\">"
            "<?DSML?parameter name=\"command\" string=\"true\">pwd"
            "</?DSML?parameter></?DSML?invoke></?DSML?tool_calls>"
        ),
    )
    code, body = run_case(server, nested, valid)
    assert code == 200, body
    call = body["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "write_file", body
    assert json.loads(call["arguments"]) == {
        "path": "/tmp/safe.py",
        "content": "print('safe')\n",
    }, body

    # A syntactically valid call that omits a required schema field gets the
    # same bounded, model-visible recovery.
    missing_path = dsml_write(path=None, content="print('unsafe')\n")
    code, body = run_case(server, missing_path, valid)
    assert code == 200, body
    call = body["choices"][0]["message"]["tool_calls"][0]["function"]
    assert json.loads(call["arguments"])["path"] == "/tmp/safe.py", body

    # If the replacement repeats the violation, fail closed with a typed
    # provider error. No tool_calls response is emitted to the harness.
    code, body = run_case(server, missing_path, missing_path)
    assert code == 422, body
    assert body["error"]["type"] == "model_output_error", body
    assert body["error"]["code"] == "invalid_tool_call", body
    assert "required property path" in body["error"]["message"], body

    # A streaming turn whose whole output was one suppressed tool block has
    # delivered nothing renderable, so it gets the same bounded, model-visible
    # recovery the atomic cases above get. This deliberately diverges from ds4,
    # which refuses recovery for every stream (ds4_server.c:11671/11754): the
    # reason ds4 gives is that emitted reasoning/content cannot be retracted,
    # which only binds once something was emitted. Finishing the turn empty is
    # itself a failure mode — an agent that sees an empty response re-sends a
    # byte-identical request and a deterministic model reproduces the same
    # malformed call.
    code, stream = run_case(server, nested, valid, stream=True)
    assert code == 200, stream
    assert "event: error" not in stream, stream
    assert '"code":"invalid_tool_call"' not in stream, stream
    assert '"finish_reason":"tool_calls"' in stream, stream
    assert "data: [DONE]" in stream, stream
    # The safety guarantee is unchanged: the rejected block is never replayed to
    # the client. Only the validated replacement reaches it.
    assert "/tmp/bad.py" not in stream, stream
    assert "partial source" not in stream, stream
    assert "/tmp/safe.py" in stream, stream

    # When the replacement repeats the violation there is nothing to recover.
    # The atomic path fails closed at a typed 422 above; a stream must NOT — it
    # drops the block and finishes "stop", because an error frame costs a
    # streaming agent the whole round (ds4_server.c:5231-5241).
    code, stream = run_case(server, missing_path, missing_path, stream=True)
    assert code == 200, stream
    assert "event: error" not in stream, stream
    assert '"code":"invalid_tool_call"' not in stream, stream
    assert '"finish_reason":"stop"' in stream, stream
    assert '"type":"function"' not in stream, stream
    assert "print('unsafe')" not in stream, stream

    # EMBER_STREAM_TOOL_RETRY=0 restores ds4's blanket refusal without a
    # rebuild, if in-stream recovery ever misbehaves in deployment.
    code, stream = run_case(
        server, nested, valid, stream=True,
        extra_env={"EMBER_STREAM_TOOL_RETRY": "0"},
    )
    assert code == 200, stream
    assert "event: error" not in stream, stream
    assert '"finish_reason":"stop"' in stream, stream
    assert "/tmp/bad.py" not in stream, stream
    assert "/tmp/safe.py" not in stream, stream
    assert '"type":"function"' not in stream, stream

    # The typed-error boundary stays reachable via EMBER_STREAM_TOOL_ERROR=1.
    # Recovery runs first, so this can only surface once recovery has failed.
    code, stream = run_case(
        server, missing_path, missing_path, stream=True,
        extra_env={"EMBER_STREAM_TOOL_ERROR": "1"},
    )
    assert code == 200, stream
    assert stream.count("event: error") == 1, stream
    assert '"code":"invalid_tool_call"' in stream, stream
    assert '"type":"function"' not in stream, stream

    # Native protocol streams take the same recovery path as chat. Responses
    # buffers the tool-bearing attempt, so the rejected block still cannot
    # escape — only the validated replacement does.
    code, stream = run_case(
        server,
        nested,
        valid,
        endpoint="/v1/responses",
        payload=responses_tool_request(),
    )
    assert code == 200, stream
    assert '"sequence_number"' in stream, stream
    assert "event: error" not in stream, stream
    assert '"type":"error"' not in stream, stream
    assert '"code":"invalid_tool_call"' not in stream, stream
    # The malformed block is recovered and the response completes normally
    # instead of erroring.
    assert '"type":"response.completed"' in stream, stream
    assert "/tmp/bad.py" not in stream, stream
    assert "partial source" not in stream, stream
    assert "/tmp/safe.py" in stream, stream

    # A malformed streaming attempt fails once, explicitly, without running the
    # configured replacement or emitting a tool-call header.
    # A complete call missing a required
    # property ("$ is missing required property ..."), which ember's schema
    # validator refuses (tool_schema.c), must finish the streaming turn
    # normally with no tool call rather than erroring.
    code, stream = run_case(
        server, missing_path, missing_path, stream=True
    )
    assert code == 200, stream
    assert "event: error" not in stream, stream
    assert '"code":"invalid_tool_call"' not in stream, stream
    assert '"type":"model_output_error"' not in stream, stream
    assert '"finish_reason":"stop"' in stream, stream
    assert '"type":"function"' not in stream, stream
    assert "data: [DONE]" in stream, stream

    # A progress-watchdog stop on a STREAMING response finishes normally, by the
    # same reasoning as the malformed-tool case above: degenerate output is model
    # output, not a server failure, and a streaming client cannot retry.
    #
    # It must still never be misread as a malformed-tool retry merely because the
    # partial/echoed prompt contains a DSML example, and
    # the reason must remain visible to a client that cares -- carried in
    # finish_details.type rather than a typed error.
    #
    # finish_reason MUST be "stop", never "length": "length" is what invites a
    # harness to request a continuation, which re-enters the same degenerate
    # decode repeatedly.
    code, stream = run_case(
        server,
        nested,
        valid,
        stream=True,
        termination_reason="prompt_echo_detected",
    )
    assert code == 200, stream
    assert "event: error" not in stream, stream
    assert '"type":"model_output_error"' not in stream, stream
    assert '"finish_reason":"stop"' in stream, stream
    assert '"finish_reason":"length"' not in stream, stream
    assert '"type":"prompt_echo_detected"' in stream, stream   # finish_details
    assert '"code":"invalid_tool_call"' not in stream, stream
    assert "/tmp/safe.py" not in stream, stream
    assert "data: [DONE]" in stream, stream

    # ── the two fallthrough paths a watchdog stop must CLAIM ─────────────────
    # A stall does not merely choose a rendering; it must own the case, or a
    # later branch handles it and emits the typed error the finish-normally
    # path exists to remove. Both branches below are reachable in deployment
    # and neither was covered when the gate was first written.
    #
    # (1) EMBER_STREAM_TOOL_ERROR=1 is a supported knob. With it set and the
    # payload carrying invalid tools, an unclaimed stall lands in the tool
    # branch and is misreported as invalid_tool_call.
    code, stream = run_case(
        server,
        nested,
        valid,
        stream=True,
        termination_reason="prompt_echo_detected",
        extra_env={"EMBER_STREAM_TOOL_ERROR": "1"},
    )
    assert code == 200, stream
    assert "event: error" not in stream, stream
    assert '"code":"invalid_tool_call"' not in stream, stream
    assert '"finish_reason":"stop"' in stream, stream
    assert '"type":"prompt_echo_detected"' in stream, stream

    # (2) A stall inside an unclosed <think> that already began a tool stanza.
    # continue_tool_started_in_think bails on any termination_reason, so the
    # bounded </think> continuation never runs and unclosed_think_tool is true
    # — an unclaimed stall is then blamed on "a tool call inside an unclosed
    # reasoning block", which is both an error frame and the wrong cause. This
    # combination became reachable precisely BECAUSE the echo rule is now
    # visible-only, so reasoning-phase stops are exactly these reasons.
    stall_think_payload = tool_request(stream=True)
    stall_think_payload["reasoning_effort"] = "high"
    stall_in_think = "reasoning before call\n" + valid
    code, stream = run_case(
        server,
        stall_in_think,
        valid,
        stream=True,
        payload=stall_think_payload,
        think_tool_reply=valid,
        termination_reason="reasoning_cycle_detected",
    )
    assert code == 200, stream
    assert "event: error" not in stream, stream
    assert "unclosed reasoning" not in stream, stream
    assert '"code":"invalid_tool_call"' not in stream, stream
    assert '"finish_reason":"stop"' in stream, stream
    assert '"type":"reasoning_cycle_detected"' in stream, stream

    # EMBER_STREAM_WATCHDOG_ERROR=1 restores the typed-error boundary without a
    # rebuild, mirroring EMBER_STREAM_TOOL_ERROR above.
    code, stream = run_case(
        server,
        nested,
        valid,
        stream=True,
        termination_reason="prompt_echo_detected",
        extra_env={"EMBER_STREAM_WATCHDOG_ERROR": "1"},
    )
    assert code == 200, stream
    assert stream.count("event: error") == 1, stream
    assert '"code":"prompt_echo_detected"' in stream, stream
    assert '"type":"model_output_error"' in stream, stream
    assert '"retry_exhausted":true' in stream, stream
    assert '"code":"invalid_tool_call"' not in stream, stream
    assert "/tmp/safe.py" not in stream, stream

    code, body = run_case(
        server,
        nested,
        valid,
        termination_reason="prompt_echo_detected",
    )
    assert code == 422, body
    assert body["error"]["type"] == "model_output_error", body
    assert body["error"]["code"] == "prompt_echo_detected", body
    assert body["usage"]["backend"]["termination_reason"] == (
        "prompt_echo_detected"
    ), body
    assert "/tmp/safe.py" not in json.dumps(body), body

    # A copied tool-preamble fragment is prompt echo, not an executable intent.
    # It can contain a complete opener before the backend accumulates the full
    # 512-token watchdog proof, so the earlier think-tool recovery boundary
    # needs its own exact-overlap guard.
    echoed_preamble = (
        f'You can invoke tools by writing a "{OPEN}tool_calls>" block like '
        f"the following:\n\n{OPEN}tool_calls>"
    )
    echo_payload = tool_request(stream=True)
    echo_payload["reasoning_effort"] = "high"
    code, stream = run_case(
        server,
        echoed_preamble,
        valid,
        stream=True,
        payload=echo_payload,
        think_tool_reply=valid,
    )
    assert code == 200, stream
    assert stream.count("event: error") == 1, stream
    assert '"code":"invalid_tool_call"' in stream, stream
    assert '"type":"function"' not in stream, stream
    assert "/tmp/safe.py" not in stream, stream

    # ds4 server parity for a common model mistake: a complete tool stanza
    # begins before </think>. Ember cannot inject from inside its backend-owned
    # eval loop, so it performs one same-assistant continuation after the first
    # decode and exposes only the post-close stanza as executable.
    in_think = "reasoning before call\n" + valid
    think_payload = tool_request(stream=True)
    think_payload["reasoning_effort"] = "high"
    code, stream = run_case(
        server,
        in_think,
        valid,
        stream=True,
        payload=think_payload,
        think_tool_reply=valid,
    )
    assert code == 200, stream
    assert "event: error" not in stream, stream
    assert stream.count('"type":"function"') == 1, stream
    assert "/tmp/safe.py" in stream, stream
    assert '"finish_reason":"tool_calls"' in stream, stream

    # The same recovery remains atomic for non-streaming callers.
    think_payload = tool_request(stream=False)
    think_payload["reasoning_effort"] = "high"
    code, body = run_case(
        server,
        in_think,
        valid,
        payload=think_payload,
        think_tool_reply=valid,
    )
    assert code == 200, body
    assert body["choices"][0]["finish_reason"] == "tool_calls", body
    assert body["choices"][0]["message"]["tool_calls"][0]["function"][
        "name"
    ] == "write_file", body

    # A quoted DSML opener is only recoverable when tools were actually
    # declared. Tool-less reasoning must not gain executable semantics or cause
    # a hidden continuation merely because it discusses the protocol syntax.
    no_tools_payload = {
        "model": "stub",
        "stream": False,
        "max_tokens": 4096,
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "explain this markup"}],
    }
    code, body = run_case(
        server,
        "quoted protocol example: " + valid,
        valid,
        payload=no_tools_payload,
        think_tool_reply="THIS CONTINUATION MUST NOT RUN",
    )
    assert code == 200, body
    assert "THIS CONTINUATION MUST NOT RUN" not in json.dumps(body), body
    assert valid in body["choices"][0]["message"]["reasoning_content"], body

    # Raw non-string parameters must be one complete JSON value. The old ds4
    # parity behavior silently replaced malformed JSON with null and allowed it
    # to cross the executable boundary.
    invalid_raw = dsml_call(
        "write_file",
        [("path", False, "not-json"), ("content", True, "unsafe")],
    )
    code, body = run_case(server, invalid_raw, invalid_raw)
    assert code == 422, body
    assert "not valid JSON" in body["error"]["message"], body

    # Duplicate names are rejected recursively before a downstream last-key
    # parser can execute a value different from Ember's first-key lookup.
    duplicate = dsml_call(
        "write_file",
        [
            ("path", True, "/tmp/first"),
            ("path", True, "/tmp/second"),
            ("content", True, "unsafe"),
        ],
    )
    code, body = run_case(server, duplicate, duplicate)
    assert code == 422, body
    assert "duplicate object keys" in body["error"]["message"], body

    # Recursive strict schema constraints cover nested types/enums and reject
    # unadvertised properties, not only top-level required-key presence.
    strict_payload = tool_request()
    strict_fn = strict_payload["tools"][0]["function"]
    strict_fn["strict"] = True
    strict_fn["parameters"] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "options": {
                "type": "object",
                "properties": {"mode": {"enum": ["safe"]}},
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
        "required": ["path", "options"],
        "additionalProperties": False,
    }
    bad_nested = dsml_call(
        "write_file",
        [
            ("path", True, "/tmp/x"),
            ("options", False, '{"mode":"unsafe","extra":1}'),
        ],
    )
    code, body = run_case(
        server, bad_nested, bad_nested, payload=strict_payload
    )
    assert code == 422, body
    assert "$.options.mode" in body["error"]["message"], body

    # Required tool choice also covers a text-only model refusal: atomic mode
    # gets one model-visible correction and then exposes only the valid call.
    required_payload = tool_request()
    required_payload["tool_choice"] = "required"
    code, body = run_case(
        server, "I will not call a tool", valid, payload=required_payload
    )
    assert code == 200, body
    assert body["choices"][0]["finish_reason"] == "tool_calls", body

    # Named and no-parallel choices are executable constraints, not prompt-only
    # suggestions.
    named_payload = tool_request()
    named_payload["tools"].append(
        {
            "type": "function",
            "function": {
                "name": "other",
                "parameters": {"type": "object", "additionalProperties": False},
            },
        }
    )
    named_payload["tool_choice"] = {
        "type": "function",
        "function": {"name": "write_file"},
    }
    wrong_named = dsml_call("other", [])
    code, body = run_case(
        server, wrong_named, wrong_named, payload=named_payload
    )
    assert code == 422, body
    assert "excluded by tool_choice" in body["error"]["message"], body

    parallel_payload = tool_request()
    parallel_payload["parallel_tool_calls"] = False
    invoke = (
        f'{OPEN}invoke name="write_file">'
        f'{OPEN}parameter name="path" string="true">/tmp/x{CLOSE}parameter>'
        f'{OPEN}parameter name="content" string="true">x{CLOSE}parameter>'
        f"{CLOSE}invoke>"
    )
    parallel = f"{OPEN}tool_calls>{invoke}{invoke}{CLOSE}tool_calls>"
    code, body = run_case(
        server, parallel, parallel, payload=parallel_payload
    )
    assert code == 422, body
    assert "parallel tool calls were disabled" in body["error"]["message"], body

    # A provider failure on the hidden replacement remains a provider failure;
    # it must never be rewritten as an invalid_tool_call 422.
    code, body = run_case(
        server,
        missing_path,
        valid,
        recovery_error="backend_retry_failed",
    )
    assert code == 500, body
    assert body["error"]["code"] == "backend_retry_failed", body

    # Stub and deployment agree that zero is a literal output budget.
    zero_payload = {
        "model": "stub",
        "messages": [{"role": "user", "content": "prewarm"}],
        "max_tokens": 0,
        "reasoning_effort": "none",
    }
    code, body = run_case(server, "abc", "unused", payload=zero_payload)
    assert code == 200, body
    assert body["choices"][0]["message"]["content"] == "", body
    assert body["usage"]["completion_tokens"] == 0, body

    # Anthropic gets Anthropic-native HTTP errors and exact stop metadata.
    code, body = run_case(
        server,
        "unused",
        "unused",
        endpoint="/v1/messages",
        payload={"model": "stub", "max_tokens": 1},
    )
    assert code == 400, body
    assert body["type"] == "error", body
    assert body["error"]["type"] == "invalid_request_error", body

    anthropic_payload = {
        "model": "stub",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "answer"}],
        "thinking": {"type": "enabled", "budget_tokens": 20},
        "stop_sequences": ["END"],
    }
    code, body = run_case(
        server,
        "private reasoning</think>helloENDtail",
        "unused",
        endpoint="/v1/messages",
        payload=anthropic_payload,
    )
    assert code == 200, body
    assert body["stop_reason"] == "stop_sequence", body
    assert body["stop_sequence"] == "END", body
    assert all(block["type"] != "thinking" for block in body["content"]), body
    assert "private reasoning" not in json.dumps(body), body

    print("tool-safety server integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
