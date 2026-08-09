#!/usr/bin/env bash
set -euo pipefail

failures=0
warnings=0

pass() { printf 'ok: %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; warnings=$((warnings + 1)); }
fail() { printf 'error: %s\n' "$*" >&2; failures=$((failures + 1)); }

if [[ "$(uname -s)" == Linux ]]; then
  pass "Linux host detected"
else
  fail "Ember's ROCm container requires a native Linux host"
fi

if [[ "$(uname -m)" == x86_64 ]]; then
  pass "x86_64 host detected"
else
  fail "the published image targets x86_64 AMD Strix Halo hosts"
fi

for device in /dev/kfd /dev/dri; do
  if [[ -r "$device" ]]; then
    pass "$device is available"
  else
    fail "$device is not readable; install/configure the ROCm-supported AMD device stack"
  fi
done

if command -v docker >/dev/null 2>&1; then
  pass "Docker CLI found"
  if docker info >/dev/null 2>&1; then
    pass "Docker daemon is reachable"
  else
    fail "Docker daemon is not reachable by the current user"
  fi
  if docker compose version >/dev/null 2>&1; then
    pass "Docker Compose plugin found"
  else
    fail "Docker Compose v2 plugin is required"
  fi
else
  fail "Docker Engine is required"
fi

memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
recommended_memory_kib=$((120 * 1024 * 1024))
if [[ "$memory_kib" =~ ^[0-9]+$ ]]; then
  if (( memory_kib >= recommended_memory_kib )); then
    pass "system memory is at least 120 GiB"
  else
    warn "less than 120 GiB of system memory is visible; the release-tested model may exhaust unified memory"
  fi
fi

models_dir="${EMBER_MODELS_DIR:-./models}"
mkdir -p "$models_dir"
available="$(df -PB1 "$models_dir" | awk 'NR == 2 {print $4}')"
required=$((94 * 1024 * 1024 * 1024))
if [[ "$available" =~ ^[0-9]+$ ]] && (( available >= required )); then
  pass "$models_dir has at least 94 GiB free"
else
  fail "$models_dir needs at least 94 GiB free for the verified model download"
fi

if command -v rocminfo >/dev/null 2>&1; then
  if rocminfo 2>/dev/null | grep -q gfx1151; then
    pass "ROCm reports a gfx1151 agent"
  else
    fail "ROCm does not report a gfx1151 (AMD Strix Halo) agent"
  fi
else
  warn "rocminfo is not installed on the host; the container will perform the architecture check"
fi

if (( failures > 0 )); then
  printf 'preflight: %d error(s), %d warning(s)\n' "$failures" "$warnings" >&2
  exit 1
fi

printf 'preflight: ready (%d warning(s))\n' "$warnings"
