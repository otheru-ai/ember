#!/usr/bin/env python3
"""End-to-end proof that omitted output limits use the model-card budget."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_request_budgets_server.py EMBER_SERVER")

    port = free_port()
    card = {
        "max_tokens": 7,
        "hard_limit_reply_budget": 2,
        "thinking_terminator_hint": "</think>\n\n",
        "reasoning_effort_tiers": {
            "low": 1,
            "medium": 2,
            "high": 3,
            "x-high": 4,
            "max": 5,
        },
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 17,
            "min_p": 0.03,
            "presence_penalty": -0.4,
            "repetition_penalty": 1.15,
        },
    }
    card_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    try:
        json.dump(card, card_file)
        card_file.close()

        env = os.environ.copy()
        env.pop("EMBER_FORCE_EXACT_PREFILL", None)
        env["EMBER_STUB_REPLY"] = "0123456789abcdef"
        env["EMBER_STUB_ECHO_SAMPLER"] = "1"
        proc = subprocess.Popen(
            [
                sys.argv[1],
                "-m",
                "stub",
                "--port",
                str(port),
                "--max-ctx",
                "4096",
                "--model-card",
                card_file.name,
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 5
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(proc.stderr.read())
                try:
                    request_json(base + "/status")
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.03)

            status_code, server_status = request_json(base + "/status")
            assert status_code == 200, server_status
            assert server_status["sampling_defaults"] == {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 17,
                "min_p": 0.03,
                "repetition_penalty": 1.15,
                "presence_penalty": -0.4,
            }, server_status

            common = {
                "model": "stub",
                "reasoning_effort": "none",
                "messages": [{"role": "user", "content": "budget proof"}],
            }
            status, omitted = request_json(
                base + "/v1/chat/completions", common
            )
            assert status == 200, omitted
            assert omitted["usage"]["completion_tokens"] == 7, omitted
            assert omitted["choices"][0]["finish_reason"] == "length", omitted

            explicit_request = dict(common)
            explicit_request["max_tokens"] = 9
            status, explicit = request_json(
                base + "/v1/chat/completions", explicit_request
            )
            assert status == 200, explicit
            assert explicit["usage"]["completion_tokens"] == 9, explicit

            sampler_request = dict(common)
            sampler_request["max_tokens"] = 256
            status, sampler = request_json(
                base + "/v1/chat/completions", sampler_request
            )
            assert status == 200, sampler
            assert sampler["choices"][0]["message"]["content"] == (
                "temp=0.7 top_p=0.9 top_k=17 min_p=0.03 rep_pen=1.15 "
                "freq_pen=0 pres_pen=-0.4 greedy=0 dry_mult=0 "
                "dry_base=0 dry_allow=-1 dry_win=0 exact_prefill=0"
            ), sampler

            overrides = dict(sampler_request)
            overrides.update(
                temperature=0,
                top_p=1,
                top_k=0,
                min_p=0,
                repetition_penalty=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            status, overridden = request_json(
                base + "/v1/chat/completions", overrides
            )
            assert status == 200, overridden
            assert overridden["choices"][0]["message"]["content"] == (
                "temp=0 top_p=1 top_k=0 min_p=0 rep_pen=1 "
                "freq_pen=0 pres_pen=0 greedy=1 dry_mult=0 "
                "dry_base=0 dry_allow=-1 dry_win=0 exact_prefill=0"
            ), overridden

            # DRY explicitly requested: proves the chat layer resolves every
            # dry_* field and forwards it across the backend ABI. Without this
            # nothing GPU-free covers main.c's resolution -- test_sampler tests
            # the math BELOW the seam and never sees a request.
            dry_req = dict(sampler_request)
            dry_req.update(
                dry_multiplier=0.8,
                dry_base=1.75,
                dry_allowed_length=3,
                dry_penalty_last_n=512,
            )
            status, dry_on = request_json(
                base + "/v1/chat/completions", dry_req
            )
            assert status == 200, dry_on
            assert dry_on["choices"][0]["message"]["content"].endswith(
                "dry_mult=0.8 dry_base=1.75 dry_allow=3 dry_win=512 "
                "exact_prefill=0"
            ), dry_on
            print("request budgets: omitted=card default, explicit=caller limit")
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        exact_port = free_port()
        exact_base = f"http://127.0.0.1:{exact_port}"
        exact_env = env.copy()
        exact_env["EMBER_FORCE_EXACT_PREFILL"] = "1"
        exact_proc = subprocess.Popen(
            [
                sys.argv[1],
                "-m",
                "stub",
                "--port",
                str(exact_port),
                "--max-ctx",
                "4096",
                "--model-card",
                card_file.name,
            ],
            env=exact_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while True:
                if exact_proc.poll() is not None:
                    raise RuntimeError(exact_proc.stderr.read())
                try:
                    request_json(exact_base + "/status")
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.03)
            exact_request = {
                "model": "stub",
                "reasoning_effort": "none",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "exact setup"}],
            }
            status, exact = request_json(
                exact_base + "/v1/chat/completions", exact_request
            )
            assert status == 200, exact
            assert exact["choices"][0]["message"]["content"].endswith(
                "exact_prefill=1"
            ), exact
        finally:
            if exact_proc.poll() is None:
                exact_proc.send_signal(signal.SIGTERM)
                try:
                    exact_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    exact_proc.kill()
                    exact_proc.wait()
    finally:
        try:
            os.unlink(card_file.name)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
