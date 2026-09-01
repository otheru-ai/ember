#!/usr/bin/env python3
"""End-to-end proof that the HTTP dispatcher admits overlapping sessions."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request_json(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_continuous_batch_server.py EMBER_SERVER")
    port = free_port()
    env = os.environ.copy()
    env["EMBER_STUB_REPLY"] = "batch-proof"
    env["EMBER_STUB_TOKEN_DELAY_US"] = "30000"
    proc = subprocess.Popen(
        [
            sys.argv[1], "-m", "stub", "--port", str(port),
            "--batch-sessions", "2", "--max-ctx", "4096",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 8
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early: {proc.stderr.read()}")
            try:
                request_json(base + "/status")
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.03)

        # Learned image blocks are larger than the resident prefill quantum.
        # The server must reject before calling the stub encoder, whose generic
        # failure would otherwise hide whether resident admission was reached.
        image_request = {
            "model": "stub",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image: "},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgo="
                    }},
                ],
            }],
        }
        image_status, image_error = request_json(
            base + "/v1/chat/completions", image_request)
        assert image_status == 400, image_error
        assert image_error["error"]["code"] == \
            "vision_resident_prefill_unsupported", image_error

        barrier = threading.Barrier(3)
        results = []
        errors = []

        def generate(prompt):
            try:
                barrier.wait()
                results.append(request_json(
                    base + "/v1/completions",
                    {"model": "stub", "prompt": prompt, "max_tokens": 11},
                ))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=generate, args=("one",)),
            threading.Thread(target=generate, args=("two",)),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(10)
        if any(thread.is_alive() for thread in threads):
            raise AssertionError("concurrent generations did not complete")
        if errors:
            raise errors[0]
        if len(results) != 2 or any(status != 200 for status, _ in results):
            raise AssertionError(f"unexpected generation results: {results}")

        status_code, status = request_json(base + "/status")
        batch = status["continuous_batching"]
        assert status_code == 200
        assert batch["enabled"] is True
        assert batch["capacity"] == 2
        assert batch["max_decode_batch"] >= 2, batch
        assert status["served"] == 2, status
        print("continuous batch server: 2 overlapping requests observed")
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
