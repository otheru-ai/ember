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
        for reason in ("context", "force_ar", "vision", "other"):
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
