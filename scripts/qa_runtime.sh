#!/usr/bin/env bash
# Runtime QA checklist (GPU-dependent half). Self-restoring window.
set -u
MODEL=/srv/lucebox/models/DeepSeek-V4-Flash-ROCMFP2-STRIX-abliterated.gguf
A=http://127.0.0.1:8090/v1/chat/completions
pass=0; fail=0
ck(){ if [ "$1" = 1 ]; then pass=$((pass+1)); echo "  ok: $2"; else fail=$((fail+1)); echo "  FAIL: $2"; fi; }
restore(){ sudo docker rm -f ember-qa >/dev/null 2>&1; sudo XDG_RUNTIME_DIR=/run/user/0 systemctl --user start lucebox-server; for i in $(seq 1 60); do curl -sf -m3 http://127.0.0.1:8000/health>/dev/null 2>&1 && { echo "[qa] lucebox restored"; return; }; sleep 5; done; }
trap restore EXIT
sudo XDG_RUNTIME_DIR=/run/user/0 systemctl --user stop lucebox-server
until ! sudo docker ps --format '{{.Names}}'|grep -q lucebox-server; do sleep 2; done
sudo docker run -d --rm --name ember-qa --network host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render -v /srv/lucebox/src:/workspace:ro \
  -v /srv/lucebox/models:/srv/lucebox/models:ro -v /srv/lucebox/ember-src:/ember \
  lucebox-rocm:7.2.4 /ember/ember-dflash --port 8090 -m "$MODEL" >/dev/null
for i in $(seq 1 60); do curl -sf -m3 http://127.0.0.1:8090/health>/dev/null 2>&1 && break; sleep 5; done

# 1. coherence
r=$(curl -s -m120 $A -H 'Content-Type: application/json' -d '{"stream":false,"max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"Capital of Japan? one word."}]}')
echo "$r" | grep -qi tokyo && ck 1 "coherence (Tokyo)" || ck 0 "coherence ($r)"

# 2. tool call (one-shot)
TOOLS='"tools":[{"type":"function","function":{"name":"get_time","description":"time","parameters":{"type":"object","properties":{}}}}]'
r=$(curl -s -m120 $A -H 'Content-Type: application/json' -d "{\"stream\":false,\"max_tokens\":128,\"temperature\":0,$TOOLS,\"messages\":[{\"role\":\"user\",\"content\":\"What time is it? Use the tool.\"}]}")
echo "$r" | grep -q "tool_calls" && ck 1 "tool call emitted" || ck 0 "tool call ($r)"

# 3. over-trigger guard: greeting + tools → no tool call
r=$(curl -s -m120 $A -H 'Content-Type: application/json' -d "{\"stream\":false,\"max_tokens\":32,\"temperature\":0,$TOOLS,\"messages\":[{\"role\":\"user\",\"content\":\"hello there\"}]}")
echo "$r" | grep -q "tool_calls" && ck 0 "over-trigger (fired on greeting)" || ck 1 "over-trigger guard (greeting → no tool)"

# 4. concurrency: 3 simultaneous → all complete (single slot serializes)
for i in 1 2 3; do curl -s -m180 $A -H 'Content-Type: application/json' -d '{"stream":false,"max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"say hi"}]}' >/tmp/qa_c$i.json & done
wait
n=$(grep -l '"content"' /tmp/qa_c1.json /tmp/qa_c2.json /tmp/qa_c3.json 2>/dev/null | wc -l)
[ "$n" = 3 ] && ck 1 "3 concurrent requests all completed" || ck 0 "concurrency ($n/3)"

# 5. degenerate: adversarial repetitive prompt → completes, no hang
r=$(timeout 120 curl -s -m110 $A -H 'Content-Type: application/json' -d '{"stream":false,"max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"a a a a a a a a a a"}]}')
echo "$r" | grep -q '"content"' && ck 1 "degenerate prompt completes (no hang)" || ck 0 "degenerate"

echo "[qa] === RUNTIME QA: $pass passed, $fail failed ==="
