#!/usr/bin/env python3
"""End-to-end proof that /metrics is served and that generations reach it.

test_metrics.c covers the aggregation itself by calling ember_metrics_render
directly. That leaves the wiring untested: the route registration and the
record_* call sites in main.c can all be deleted without failing a single test,
because nothing asks the running server for its metrics. A refactor did exactly
that -- dropping the route, the queue-wait probe and the prefix-cache probe --
and the whole suite stayed green. This test closes that gap over real HTTP.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers, response.read().decode()


def post_json(url: str, body: dict):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


def series(text: str, name: str) -> float:
    """Value of a bare (unlabelled) sample, ignoring HELP/TYPE lines."""
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        if key == name:
            return float(value)
    raise AssertionError(f"series {name} absent from exposition:\n{text}")


@contextmanager
def timing_server(server, overrides, batch_sessions=1):
    port = free_port()
    env = os.environ.copy()
    env.update(overrides)
    proc = subprocess.Popen(
        [server, "-m", "stub", "--port", str(port), "--max-ctx", "4096",
         "--batch-sessions", str(batch_sessions)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 5
        while True:
            assert proc.poll() is None, "timing server exited"
            try:
                get(base + "/status")
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def timing_regressions(server):
    payload = {"model": "stub", "messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 12, "reasoning_effort": "none"}
    # Occupy all dispatch workers before enqueueing one more request. This
    # distinguishes FIFO time from the old post-dequeue mutex measurement,
    # and checks that batched requests are also observed.
    for workers in (1, 2):
        with timing_server(server, {"EMBER_STUB_REPLY": "abcdefghijkl",
                                   "EMBER_STUB_TOKEN_DELAY_US": "50000"}, workers) as base:
            with ThreadPoolExecutor(max_workers=workers + 1) as pool:
                active = [pool.submit(post_json, base + "/v1/chat/completions", payload)
                          for _ in range(workers)]
                deadline = time.monotonic() + 3
                while json.loads(get(base + "/status")[2])["busy"] < workers:
                    assert time.monotonic() < deadline, "workers never became busy"
                    time.sleep(0.005)
                post_json(base + "/v1/chat/completions", payload)
                for future in active:
                    future.result()
            body = get(base + "/metrics")[2]
            assert series(body, "ember_queue_seconds_count") == workers + 1, body
            assert series(body, "ember_queue_seconds_sum") > 0.25, body

    # A long, hidden malformed attempt followed by a short valid replacement.
    # Recovery must not move TTFT to the replacement's first token. An empty
    # replacement must also retain the original callback observations.
    from test_tool_safety_server import dsml_write, tool_request
    initial = dsml_write(path=None, content="x" * 120)
    valid = dsml_write(path="/tmp/metrics", content="ok")
    for initial_reply, replacement, thinking in ((initial, valid, False),
                                                (initial, "", False),
                                                ("reasoning " * 20 + valid, valid, True)):
        with timing_server(server, {"EMBER_STUB_REPLY": initial_reply,
                                   "EMBER_STUB_RECOVERY_REPLY": replacement,
                                   "EMBER_STUB_THINK_TOOL_REPLY": replacement,
                                   "EMBER_STUB_TOKEN_DELAY_US": "3000"}) as base:
            payload = tool_request()
            if thinking:
                payload["reasoning_effort"] = "high"
            request = urllib.request.Request(
                base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    completion = json.load(response)
                    calls = completion["choices"][0]["message"].get("tool_calls", [])
                    if replacement:
                        assert calls, completion
                        assert json.loads(calls[0]["function"]["arguments"])["path"] == "/tmp/metrics", completion
                    else:
                        assert not calls, completion
            except urllib.error.HTTPError as error:
                assert replacement == "" and error.code == 422, error
                error.read()
            body = get(base + "/metrics")[2]
            assert series(body, "ember_time_to_first_token_seconds_count") == 1, body
            assert series(body, "ember_time_to_first_token_seconds_sum") < 0.4, body
            assert series(body, "ember_request_mean_token_gap_seconds_count") == 1, body


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_metrics_server.py EMBER_SERVER")

    port = free_port()
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "metrics probe"
    proc = subprocess.Popen(
        [sys.argv[1], "-m", "stub", "--port", str(port), "--max-ctx", "4096"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 5
        while True:
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read())
            try:
                get(base + "/status")
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.03)

        status, headers, body = get(base + "/metrics")
        assert status == 200, status
        # Scrapers content-negotiate on the exposition version; a bare
        # text/plain would still 200 here but breaks real Prometheus.
        assert "version=0.0.4" in headers["Content-Type"], headers["Content-Type"]
        assert "# TYPE ember_generations_total counter" in body, body

        before = series(body, "ember_generations_total")

        status, completion = post_json(base + "/v1/chat/completions", {
            "model": "stub",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        })
        assert status == 200, completion

        _, _, body = get(base + "/metrics")
        after = series(body, "ember_generations_total")
        # The route alone proves nothing: without the record_generation call
        # site the counter stays flat while every other assertion still passes.
        assert after == before + 1, f"{before} -> {after}\n{body}"
        assert series(body, "ember_prefix_cache_requests_total") >= 1.0, body

        # #9's latency gap: TTFT is queue + prefill and must be its own series.
        # Asserting it advanced with the generation is what distinguishes a
        # wired-up histogram from a declared-but-never-observed one.
        assert series(body, "ember_time_to_first_token_seconds_count") >= 1.0, body
        # Present even at zero. A missing series and a zero one look identical
        # to a human reading a dashboard, but not to a scraper building a graph.
        for name in ("ember_spec_decode_declined_total",
                     "ember_request_mean_token_gap_seconds_count",
                     "ember_vision_encoder_seconds_count",
                     "ember_image_tokens_total",
                     "ember_request_image_count_count"):
            assert name in body, f"{name} absent from the exposition:\n{body}"
        # The closed label set must be exported in full, so a reason that has
        # not occurred yet still graphs as zero rather than appearing later and
        # looking like a spike.
        for reason in ("context", "force_ar", "vision", "other",
                       "resident_provider", "resident_submit_failed",
                       "resident_shadow_capture"):
            assert f'ember_spec_decode_declined_total{{reason="{reason}"}}' in body, \
                f"decline reason {reason} not exported"

        # /status advertises modalities, llama.cpp /props parity. The stub has
        # no tower, so vision MUST be false here -- a true value would mean the
        # capability is being read from the architecture rather than from what
        # the operator actually supplied.
        status_code, _, status_body = get(base + "/status")
        assert status_code == 200, status_code
        modalities = json.loads(status_body).get("modalities")
        assert modalities == {"text": True, "vision": False,
                              "vision_scope": "configured"}, modalities

        print("metrics server ok")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
    timing_regressions(sys.argv[1])
