#!/usr/bin/env bash
set -euo pipefail

base_url="${EMBER_BASE_URL:-http://127.0.0.1:8080}"
generate=0

if [[ "${1:-}" == --generate ]]; then
  generate=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--generate]" >&2
  exit 64
fi

request() {
  curl --fail --silent --show-error --max-time "${EMBER_SMOKE_TIMEOUT:-30}" "$@"
}

request "$base_url/health" >/dev/null
echo "ok: health"

models="$(request "$base_url/v1/models")"
python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("object") == "list" and d.get("data")' <<<"$models"
echo "ok: model discovery"

status="$(request "$base_url/status")"
python3 -c 'import json,sys; assert isinstance(json.load(sys.stdin), dict)' <<<"$status"
echo "ok: status"

if (( generate )); then
  response="$(request \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Reply with only: ember-ready"}],"max_tokens":16,"temperature":0,"stream":false}' \
    "$base_url/v1/chat/completions")"
  python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["choices"][0]["message"]["content"]' <<<"$response"
  echo "ok: generation"
fi

echo "smoke test passed: $base_url"
