#!/usr/bin/env python3
"""GPU-free build-contract and ABI smoke tests for the llama.cpp provider."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_qwen_vision_provider.sh"
CHECK = ROOT / "scripts" / "check_qwen_vision_provider.py"


class QwenVisionProviderScriptTests(unittest.TestCase):
    def test_dry_run_pins_exact_llama_revision_and_backend(self) -> None:
        result = subprocess.run(
            [str(BUILD), "--build-dir", "/tmp/build", "--install-dir", "/tmp/install",
             "--backend", "hip", "--dry-run"],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ref=refs/pull/27774/head", result.stdout)
        self.assertIn("revision=abdc7a0bf815d3b83e26dd523c6960e4dd597e82", result.stdout)
        self.assertIn("backend=hip", result.stdout)

    def test_checker_accepts_complete_v1_table_and_rejects_symlink(self) -> None:
        source = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
struct output { uint32_t t, h, w, width; size_t rows; float * data; };
struct api {
    uint32_t version;
    void * (*create)(const char *, const char *, int, char *, size_t);
    void (*destroy)(void *);
    bool (*encode)(void *, const uint8_t *, size_t, struct output *, char *, size_t);
    void (*free_output)(void *, struct output *);
};
static void * make(const char * a, const char * b, int c, char * d, size_t e) {
    (void)a; (void)b; (void)c; (void)d; (void)e; return (void *)1;
}
static void drop(void * p) { (void)p; }
static bool encode(void * p, const uint8_t * b, size_t n, struct output * o,
                   char * e, size_t z) {
    (void)p; (void)b; (void)n; (void)o; (void)e; (void)z; return false;
}
static void release(void * p, struct output * o) { (void)p; (void)o; }
static const struct api table = {1, make, drop, encode, release};
__attribute__((visibility("default")))
const struct api * qwen4exp_vision_provider_get_v1(void) { return &table; }
"""
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            c_file = tmp / "provider.c"
            provider = tmp / "provider.so"
            c_file.write_text(source, encoding="utf-8")
            subprocess.run(
                ["cc", "-shared", "-fPIC", "-fvisibility=hidden", str(c_file),
                 "-o", str(provider)], check=True,
            )
            accepted = subprocess.run(
                [sys.executable, str(CHECK), "--provider", str(provider)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            link = tmp / "provider-link.so"
            link.symlink_to(provider)
            rejected = subprocess.run(
                [sys.executable, str(CHECK), "--provider", str(link)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("must not be a symlink", rejected.stderr)

    def test_adapter_uses_vocab_only_text_view(self) -> None:
        source = (ROOT / "tools" / "qwen4exp_vision_provider_llamacpp.cpp").read_text()
        self.assertIn("model_params.vocab_only = true", source)
        self.assertIn("params.warmup = false", source)
        self.assertNotIn("llama_new_context_with_model", source)


if __name__ == "__main__":
    unittest.main()
