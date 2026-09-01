#!/usr/bin/env bash
set -u
MODEL=/srv/lucebox/models/DeepSeek-V4-Flash-ROCMFP2-STRIX-abliterated.gguf
KV=/srv/lucebox/ember-kvcache
run_ember() {
  sudo docker run -d --rm --name ember-c --network host \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    -v /srv/lucebox/src:/workspace:ro -v /srv/lucebox/models:/srv/lucebox/models:ro \
    -v /srv/lucebox/ember-src:/ember -v "$KV":/kv \
    lucebox-rocm:7.2.4 /ember/ember-dflash --port 8090 -m "$MODEL" --kv-cache-dir /kv >/dev/null
  for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8090/health >/dev/null 2>&1 && return 0; sudo docker ps --format '{{.Names}}'|grep -q ember-c || return 1; sleep 5; done; return 1
}
restore() {
  sudo docker rm -f ember-c >/dev/null 2>&1
  sudo XDG_RUNTIME_DIR=/run/user/0 systemctl --user start lucebox-server
  for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "[c] lucebox restored"; return; }; sleep 5; done
}
trap restore EXIT
sudo XDG_RUNTIME_DIR=/run/user/0 systemctl --user stop lucebox-server
until ! sudo docker ps --format '{{.Names}}' | grep -q lucebox-server; do sleep 2; done

python3 - <<'PY'
import json
sysp=" ".join("Rule %d: verify subsystem %d." % (i,i%40) for i in range(650))
open('/tmp/creq.json','w').write(json.dumps({"model":"deepseek-v4-flash","stream":False,"max_tokens":12,"temperature":0,
  "messages":[{"role":"system","content":sysp},{"role":"user","content":"Reply: ok"}]}))
open('/tmp/creqs.json','w').write(json.dumps({"model":"deepseek-v4-flash","stream":True,"max_tokens":24,"temperature":0,
  "messages":[{"role":"user","content":"Count 1 to 3."}]}))
PY

echo "[c] ===== run 1 ====="
run_ember || { echo "[c] failed to start"; sudo docker logs ember-c 2>&1|tail; exit 1; }
echo "[c] --- TEST A: streaming through real backend ---"
curl -s -N -m 120 http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/creqs.json \
 | grep -a "^data:" | head -8 | sed 's/^data: //' | python3 -c "
import json,sys
c=''
for l in sys.stdin:
    l=l.strip()
    if not l or l=='[DONE]': continue
    try: c+=json.loads(l)['choices'][0].get('delta',{}).get('content') or ''
    except: pass
print('  streamed content:',repr(c[:80]))"
echo "[c] --- write disk snapshot (cold) ---"
c1=$( { /usr/bin/time -f "%e" curl -s -m 300 http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/creq.json >/dev/null; } 2>&1 )
echo "  turn1 wall=${c1}s ; disk files: $(sudo find "$KV" -type f | wc -l)"
sudo docker rm -f ember-c >/dev/null 2>&1; sleep 2

echo "[c] ===== run 2 (fresh process — in-memory cache empty; must hit DISK) ====="
run_ember || { echo "[c] restart failed"; exit 1; }
c2=$( { /usr/bin/time -f "%e" curl -s -m 300 http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/creq.json >/dev/null; } 2>&1 )
echo "[c] === CROSS-RESTART: cold=${c1}s  fresh-process-disk-hit=${c2}s ==="
