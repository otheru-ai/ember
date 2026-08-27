#!/usr/bin/env python3
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qwen_snapshot_fetch.py"


class Handler(BaseHTTPRequestHandler):
    files = {"small.txt": b"small payload", "weights.bin": bytes(range(251)) * 1000}

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        body = self.files[name]
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *_args):
        pass


def blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rows = []
        for name, data in Handler.files.items():
            row = {"path": name, "size": len(data), "git_blob": blob(data)}
            if name.endswith(".bin"):
                row["sha256"] = hashlib.sha256(data).hexdigest()
            rows.append(row)
        self.inventory = self.root / "inventory.json"
        self.inventory.write_text(json.dumps({
            "repo_id": "fixture/model", "revision": "a" * 40,
            "file_count": len(rows), "total_bytes": sum(x["size"] for x in rows),
            "files": rows,
        }))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def run_fetch(self, *extra):
        return subprocess.run([
            "python3", str(SCRIPT), "--inventory", str(self.inventory),
            "--output", str(self.root / "out"), "--base-url",
            f"http://127.0.0.1:{self.server.server_port}", *extra,
        ], text=True, capture_output=True)

    def test_plan_is_read_only(self):
        result = self.run_fetch("--plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "out").exists())
        self.assertEqual(json.loads(result.stdout)["file_count"], 2)

    def test_download_resume_and_cache(self):
        out = self.root / "out"
        out.mkdir()
        partial = out / ".weights.bin.partial"
        partial.write_bytes(Handler.files["weights.bin"][:12345])
        result = self.run_fetch("--workers", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((out / "weights.bin").read_bytes(), Handler.files["weights.bin"])
        again = self.run_fetch()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn('"downloaded_files": 0', again.stdout)

    def test_rejects_unsafe_inventory(self):
        data = json.loads(self.inventory.read_text())
        data["files"][0]["path"] = "../escape"
        self.inventory.write_text(json.dumps(data))
        result = self.run_fetch("--plan")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe inventory path", result.stderr)


if __name__ == "__main__":
    unittest.main()
