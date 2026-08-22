# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read [`AGENTS.md`](AGENTS.md) — it is the single source of truth for agents
working in this repository, and it assumes no prior knowledge of the project.**

This file used to carry its own condensed copy of that guidance. Two documents
with the same headings and diverging bodies drifted apart (the copy here had
gone stale on `main.c`'s size and on which tests compile engine sources
directly), so the copy was removed rather than maintained twice. Add new
guidance to `AGENTS.md`.

`AGENTS.md` covers the build and test commands, the repository layout, the
architectural invariants you must not break (the backend ABI seam, persistent
generation workers, request-pipeline ordering, buffer-and-resplit SSE, KV cache
key derivation, exact-token tool replay), the coding conventions, and the
container/security boundaries.

Supporting documents it points at: [`README.md`](README.md) for installation
and first use, [`ARCHITECTURE.md`](ARCHITECTURE.md) for the layering rationale,
[`engine/VENDOR.md`](engine/VENDOR.md) for vendored-fork provenance, and
[`docs/`](docs/) for the design and audit notes.
