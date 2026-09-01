#!/usr/bin/env bash
# Prove ember-dflash emits a real token end-to-end. Briefly stops lucebox
# (Hermes offline ~3-6 min), runs ember-dflash on the freed GPU with the
# abliterated model, sends one request, then ALWAYS restores lucebox.
set -u
MODEL=/srv/lucebox/models/DeepSeek-V4-Flash-ROCMFP2-STRIX-abliterated.gguf
B=/srv/lucebox/src/server/build-rocm724
GGML="$B/deps/llama.cpp/ggml/src:$B/deps/llama.cpp/ggml/src/ggml-hip:$B/deps/llama.cpp/ggml/src/ggml-cpu"

restore() {
  echo "[proof] restoring lucebox production…"
  sudo docker rm -f ember-dflash-test >/dev/null 2>&1
  sudo XDG_RUNTIME_DIR=/run/user/0 systemctl --user start lucebox-server
  for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "[proof] lucebox restored"; return; }; sleep 5; done
  echo "[proof] WARNING: lucebox not healthy after 5min — check manually"
}
trap restore EXIT

echo "[proof] stopping lucebox (Hermes offline for the window)…"
sudo XDG_RUNTIME_DIR=/run/user/0 systemctl --user stop lucebox-server
until ! sudo docker ps --format '{{.Names}}' | grep -q lucebox-server; do sleep 2; done

echo "[proof] launching ember-dflash on :8090…"
sudo docker run -d --rm --name ember-dflash-test --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v /srv/lucebox/src:/workspace:ro -v /srv/lucebox/models:/srv/lucebox/models:ro -v /srv/lucebox/ember-src:/ember \
  -e DFLASH_DS4_SPEC=1 -e DFLASH_DS4_DRAFT=/srv/lucebox/models/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf \
  -e DFLASH_DS4_FUSED_VERIFY=1 -e DFLASH_DS4_SPEC_Q=4 -e LUCE_MMVQ_MAX_NCOLS=4 \
  -e LD_LIBRARY_PATH="$GGML:/opt/rocm/lib" \
  lucebox-rocm:7.2.4 /ember/ember-dflash --port 8090 -m "$MODEL" --model-name deepseek-v4-flash

echo "[proof] waiting for model load (up to 4 min)…"
ok=0
for i in $(seq 1 48); do
  curl -sf -m 3 http://127.0.0.1:8090/health >/dev/null 2>&1 && { ok=1; break; }
  sudo docker ps --format '{{.Names}}' | grep -q ember-dflash-test || { echo "[proof] container died:"; sudo docker logs ember-dflash-test 2>&1 | tail -20; break; }
  sleep 5
done
if [ "$ok" = 1 ]; then
  echo "[proof] === FIRST REAL TOKEN TEST ==="
  curl -s -m 120 http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","stream":false,"max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"Reply with exactly: Ember lives."}]}'
  echo; echo "[proof] === /status ==="; curl -s -m 5 http://127.0.0.1:8090/status; echo
fi
