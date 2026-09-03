#!/usr/bin/env python3
"""GPU-free cover for the benchmark's vision group.

The vision measurement runs only on the gfx1151 box, after production has been
quiesced, at the end of a group that takes several minutes to reach. A mistake
there is therefore expensive to find and slow to retry -- a missing `import
sys` cost a full box window on 2026-09-03, because the NameError landed exactly
where the group starts and the bundle script sends the harness's stderr to
/dev/null.

So this drives run_vision_group against a stub HTTP server in-process. It
proves the request carries the image, that the summary separates the cold
request from the warm ones, and that a refusing server is reported rather than
raised.
"""

from __future__ import annotations

import base64
import http.server
import importlib.util
import json
import pathlib
import threading
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bench", ROOT / "scripts" / "bench" / "benchmark.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

PROBE = ROOT / "share" / "bench" / "vision-probe.png"


class Handler(http.server.BaseHTTPRequestHandler):
    """Answers like the server does, and records what it was sent."""

    received: list[dict] = []
    fail_with: int | None = None

    def log_message(self, *args) -> None:  # keep the test output readable
        pass

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Handler.received.append(json.loads(body))
        if Handler.fail_with is not None:
            self.send_response(Handler.fail_with)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"refused"}}')
            return
        n = len(Handler.received)
        payload = {
            "choices": [{"message": {"content": "a picture"},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1049,
                "completion_tokens": 32,
                # The first request pays the tower load, so it is slower on
                # purpose: the summary must not average that away.
                "timings": {"prefill_ms": 9000.0 if n == 1 else 1800.0,
                            "decode_ms": 1450.0,
                            "prefill_tokens": 1049},
                "backend": {"spec_ran": False, "prefill_mode": "exact"},
            },
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class VisionBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        Handler.received = []
        Handler.fail_with = None
        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        port = self.server.server_address[1]
        self.endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.out = ROOT / "build-test-vision.jsonl"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.out.unlink(missing_ok=True)

    def suite(self) -> "bench.Suite":
        return bench.Suite(self.endpoint, self.out, 30.0, "stub")

    def test_probe_image_is_a_real_png(self) -> None:
        raw = PROBE.read_bytes()
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(raw), 1000, "a near-empty image would not "
                                           "exercise decode cost")

    def test_request_carries_the_image_and_the_question(self) -> None:
        bench.run_vision_group(self.suite(), PROBE, 2, 32)
        self.assertEqual(len(Handler.received), 2)
        content = Handler.received[0]["messages"][0]["content"]
        self.assertIsInstance(content, list, "an image makes content an array")
        kinds = [part["type"] for part in content]
        self.assertEqual(kinds, ["image_url", "text"])
        url = content[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]),
                         PROBE.read_bytes(),
                         "the server must receive the probe bytes unchanged")

    def test_cold_request_is_reported_separately_from_warm(self) -> None:
        result = bench.run_vision_group(self.suite(), PROBE, 4, 32)
        self.assertEqual(result["samples"], 4)
        # The stub makes the first request 5x slower. If the summary averaged
        # it in, warm_prefill_ms would be dragged well above 1800.
        self.assertEqual(result["cold_prefill_ms"], 9000.0)
        self.assertEqual(result["warm_prefill_ms"], 1800.0)
        self.assertEqual(result["image_sha256"],
                         bench.sha256_bytes(PROBE.read_bytes()))
        self.assertEqual(result["prompt_tokens"], 1049)
        self.assertTrue(all(v is False for v in result["spec_ran"]),
                        "image requests decline speculation; record it")

    def test_a_refusing_server_is_reported_not_raised(self) -> None:
        Handler.fail_with = 422
        result = bench.run_vision_group(self.suite(), PROBE, 2, 32)
        self.assertEqual(result["samples"], 0)
        self.assertIn("error", result)

    def test_text_requests_keep_the_plain_string_content(self) -> None:
        # The image path must not change the shape of an ordinary request, or
        # every text number in the bundle would move for an unrelated reason.
        self.suite().request("t", "hello", 8, group="g", repeat=1)
        self.assertEqual(Handler.received[-1]["messages"][0]["content"], "hello")


if __name__ == "__main__":
    unittest.main(verbosity=0)
