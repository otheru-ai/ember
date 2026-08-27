#!/usr/bin/env python3
"""GPU-free tests for the Qwen4Exp performance baseline recorder."""

from __future__ import annotations

import hashlib
import http.server
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_baseline.py"
SHA = "a" * 64


class Handler(http.server.BaseHTTPRequestHandler):
    payload = None

    def do_POST(self):  # noqa: N802
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        Handler.payload = json.loads(raw)
        body = json.dumps({
            "usage": {
                "prompt_tokens": 239900,
                "completion_tokens": 128,
                "restored_prefix": 0,
                "timings": {
                    "prefill_tokens": 239900,
                    "prefill_ms": 1200.5,
                    "prefill_tokens_per_sec": 199833.4,
                    "decode_ms": 6400.0,
                    "decode_tokens_per_sec": 20.0,
                },
                "backend": {
                    "prefill_mode": "exact-q1",
                    "prefill_reason": "qwen4exp_text_runtime",
                },
            },
            "choices": [{"finish_reason": "length"}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class BaselineTests(unittest.TestCase):
    def base(self, root: pathlib.Path) -> list[str]:
        return [
            sys.executable, str(SCRIPT),
            "--model", str(root / "model.gguf"),
            "--model-sha256", SHA,
            "--server-artifact", str(root / "ember-dflash"),
            "--server-pid", "123",
            "--out", str(root / "record.json"),
        ]

    def test_dry_run_touches_no_input_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            result = subprocess.run(
                [*self.base(root), "--dry-run"], text=True,
                capture_output=True, check=True,
            )
            plan = json.loads(result.stdout)
            self.assertEqual(plan["mode"], "dry-run")
            self.assertEqual(plan["plan"]["native_context"], 262144)
            self.assertEqual(plan["plan"]["implementation_path"],
                             "cpu_orchestrated_q1")
            self.assertIn("future path", plan["plan"]["path_labels"]["fused_hip"])

    def test_invalid_native_shape_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            result = subprocess.run(
                [*self.base(root), "--dry-run", "--target-prompt-tokens", "262100",
                 "--decode-tokens", "128"], text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fit native context", result.stderr)

    def test_real_fused_label_cannot_be_spoofed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (root / "model.gguf").write_bytes(b"model")
            (root / "ember-dflash").write_bytes(b"server")
            result = subprocess.run(
                [*self.base(root), "--implementation-path", "fused_hip"],
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("authoritative execution path", result.stderr)

    def test_record_captures_provenance_timings_and_resource_peaks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            model = root / "model.gguf"
            artifact = root / "ember-dflash"
            model.write_bytes(b"model")
            artifact.write_bytes(b"artifact")
            proc = root / "proc" / "123"
            proc.mkdir(parents=True)
            (proc / "status").write_text("VmRSS:\t1024 kB\nVmHWM:\t2048 kB\n")
            gtt = root / "gtt"
            gtt.write_text("3145728\n")
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            try:
                subprocess.run(
                    [*self.base(root), "--endpoint", endpoint,
                     "--proc-root", str(root / "proc"),
                     "--gtt-path", str(gtt), "--decode-tokens", "128"],
                    text=True, capture_output=True, check=True,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            record = json.loads((root / "record.json").read_text())
            self.assertEqual(record["server_artifact"]["sha256"],
                             hashlib.sha256(b"artifact").hexdigest())
            self.assertEqual(record["model_artifacts"]["files"][0]["sha256"], SHA)
            self.assertEqual(record["model_artifacts"]["aggregate_bytes"], 5)
            self.assertEqual(record["resources"]["peak_rss_bytes"], 1024 * 1024)
            self.assertEqual(record["resources"]["peak_gtt_bytes"], 3145728)
            self.assertEqual(record["phase"]["prefill_ms"], 1200.5)
            self.assertEqual(record["phase"]["decode_ms"], 6400.0)
            self.assertEqual(record["phase"]["backend_prefill_mode"], "exact-q1")
            self.assertEqual(Handler.payload["max_tokens"], 128)

    def test_split_manifest_records_complete_ordered_set(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            artifact = root / "ember-dflash"
            artifact.write_bytes(b"artifact")
            names = ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]
            entries = []
            for index, name in enumerate(names, 1):
                path = root / name
                data = f"shard-{index}".encode()
                path.write_bytes(data)
                entries.append({"filename": name, "size_bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()})
            manifest = root / "artifact-manifest.json"
            manifest.write_text(json.dumps({"artifacts": entries}))
            proc = root / "proc" / "123"
            proc.mkdir(parents=True)
            (proc / "status").write_text("VmRSS:\t1 kB\n")
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                subprocess.run([
                    sys.executable, str(SCRIPT),
                    "--artifact-manifest", str(manifest),
                    "--server-artifact", str(artifact),
                    "--server-pid", "123", "--out", str(root / "record.json"),
                    "--proc-root", str(root / "proc"),
                    "--endpoint", f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                ], text=True, capture_output=True, check=True)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)
            record = json.loads((root / "record.json").read_text())
            models = record["model_artifacts"]
            self.assertEqual(models["source"], "release-artifact-manifest")
            self.assertEqual([pathlib.Path(item["path"]).name for item in models["files"]], names)
            self.assertEqual(models["aggregate_bytes"], 14)


if __name__ == "__main__":
    unittest.main()
