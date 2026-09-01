#!/usr/bin/env bash
set -u
MODEL=/srv/lucebox/models/DeepSeek-V4-Flash-ROCMFP2-STRIX-abliterated.gguf
restore() {
  sudo docker rm -f ember-kv >/dev/null 2>&1
  sudo XDG_RUNTIME_DIR=/run/user/0 systemctl --user start lucebox-server
  for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "[kv] lucebox restored"; return; }; sleep 5; done
}
trap restore EXIT
sudo XDG_RUNTIME_DIR=/run/user/0 systemctl --user stop lucebox-server
until ! sudo docker ps --format '{{.Names}}' | grep -q lucebox-server; do sleep 2; done
sudo docker run -d --rm --name ember-kv --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v /srv/lucebox/src:/workspace:ro -v /srv/lucebox/models:/srv/lucebox/models:ro -v /srv/lucebox/ember-src:/ember \
  lucebox-rocm:7.2.4 /ember/ember-dflash --port 8090 -m "$MODEL"
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8090/health >/dev/null 2>&1 && break; sudo docker ps --format '{{.Names}}'|grep -q ember-kv || { echo "[kv] died"; sudo docker logs ember-kv 2>&1|tail; exit 1; }; sleep 5; done
# large shared system prompt + short question; send TWICE. Turn 2 should reuse
# the cached system-prompt prefix (anchor) → much faster.
python3 - <<'PY'
import json
sys=" ".join("Directive %d: operators verify gauge %d before cycle." % (i,i%50) for i in range(700))
open('/tmp/kvreq.json','w').write(json.dumps({
  "model":"deepseek-v4-flash","stream":False,"max_tokens":16,"temperature":0,
  "messages":[{"role":"system","content":sys},{"role":"user","content":"Reply: ok"}]}))
PY
echo "[kv] === TURN 1 (cold) ==="
t1=$( { /usr/bin/time -f "%e" curl -s -m 300 http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/kvreq.json >/tmp/r1.json; } 2>&1 )
echo "  wall=${t1}s  reply=$(python3 -c "import json;print(json.load(open('/tmp/r1.json'))['choices'][0]['message']['content'][:40])" 2>/dev/null)"
echo "[kv] === TURN 2 (identical prompt → KV reuse) ==="
t2=$( { /usr/bin/time -f "%e" curl -s -m 300 http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/kvreq.json >/tmp/r2.json; } 2>&1 )
echo "  wall=${t2}s"
echo "[kv] === speedup: turn1=${t1}s turn2=${t2}s ==="
