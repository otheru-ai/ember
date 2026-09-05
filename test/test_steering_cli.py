#!/usr/bin/env python3
"""Exercise startup parsing and ABI validation through the actual stub server."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prompt = root / "prompt.txt"
        prompt.write_text("Say hello")
        direction = root / "direction.bin"
        direction.write_bytes(struct.pack("<f", 1.0) + bytes(43 * 4096 * 4 - 4))
        bad = root / "truncated.bin"
        bad.write_bytes(bytes(43 * 4096 * 4 - 1))
        base = [sys.argv[1], "-m", "stub", "--validate-prompt", str(prompt),
                "--validate-tokens", "4"]
        for options in [[], ["--dir-steering-file", str(direction)],
                        ["--dir-steering-file", str(direction), "--dir-steering-ffn", "0"],
                        ["--dir-steering-file", str(direction), "--dir-steering-ffn", "3.5"],
                        ["--dir-steering-attn", "-1", "--dir-steering-file", str(direction)],
                        ["--dir-steering-attn", "0", "--dir-steering-ffn", "0"]]:
            result = subprocess.run(base + options, capture_output=True, text=True, timeout=5)
            assert result.returncode == 0, (options, result.stderr)
            assert json.loads(result.stdout)["ok"], result.stdout
        for options in [["--dir-steering-file"], ["--dir-steering-file", ""],
                        ["--dir-steering-file", str(root / "missing")],
                        ["--dir-steering-file", str(bad), "--dir-steering-ffn", "0"],
                        ["--dir-steering-attn", "1"], ["--dir-steering-ffn", "-1"]]:
            result = subprocess.run(base + options, capture_output=True, text=True, timeout=5)
            assert result.returncode != 0, options
            assert "steering" in result.stderr or "direction" in result.stderr, result.stderr
        for flag in ["--dir-steering-attn", "--dir-steering-ffn"]:
            for value in ["nan", "inf", "-inf", "101", "-101", "3.5junk", ""]:
                result = subprocess.run(base + [flag, value], capture_output=True,
                                        text=True, timeout=5)
                assert result.returncode == 2 and "invalid value" in result.stderr, (flag, value)
    print("steering CLI startup cases pass (stub structural validation only)")


if __name__ == "__main__":
    main()
