#!/usr/bin/env python3
"""GPU-free ABI and ELF dependency smoke check for the Qwen vision plugin."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import re
import subprocess
import sys


class ProviderApi(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("create", ctypes.c_void_p),
        ("destroy", ctypes.c_void_p),
        ("encode", ctypes.c_void_p),
        ("free_output", ctypes.c_void_p),
    ]


def command(*args: str) -> str:
    result = subprocess.run(
        args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{' '.join(args)} failed:\n{result.stdout}")
    return result.stdout


def check(path: Path, require_llama_deps: bool) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"provider path must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"provider is not a regular file: {path}")
    symbols = command("readelf", "--wide", "--dyn-syms", str(path))
    exported = re.findall(r"\bqwen4exp_vision_provider_get_v1\b", symbols)
    if len(exported) != 1:
        raise ValueError("provider must export qwen4exp_vision_provider_get_v1 exactly once")
    dynamic = command("readelf", "--wide", "--dynamic", str(path))
    linkage = command("ldd", str(path))
    if "not found" in linkage:
        raise ValueError(f"provider has unresolved dependencies:\n{linkage}")
    if require_llama_deps:
        for dependency in ("libmtmd.so", "libllama.so", "libggml"):
            if dependency not in linkage:
                raise ValueError(f"provider dependency closure lacks {dependency}")
        if "$ORIGIN" not in dynamic:
            raise ValueError("provider must use an $ORIGIN RUNPATH for its private closure")

    library = ctypes.CDLL(str(path), mode=getattr(ctypes, "RTLD_LOCAL", 0))
    getter = library.qwen4exp_vision_provider_get_v1
    getter.argtypes = []
    getter.restype = ctypes.POINTER(ProviderApi)
    api = getter().contents
    if api.abi_version != 1 or not all(
        (api.create, api.destroy, api.encode, api.free_output)
    ):
        raise ValueError("provider returned an incomplete or incompatible v1 table")
    return {
        "provider": str(path),
        "abi_version": api.abi_version,
        "llama_dependencies_required": require_llama_deps,
        "resolved": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--require-llama-deps", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = check(args.provider, args.require_llama_deps)
    except (OSError, ValueError) as exc:
        print(f"check_qwen_vision_provider.py: error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Qwen vision provider ABI v{report['abi_version']} and dependency closure OK: "
        f"{report['provider']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
