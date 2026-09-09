#!/usr/bin/env python3
"""Exercise the production worker loop with a fake model, without HIP.

Extract the worker/control bodies and call structs from backend_dflash.cc;
compile them with the real scheduler, executor and resident coordinator. Only
model access, telemetry and a deterministic pump barrier are substituted. This
tests the actual waits/lifetime code, unlike the wake predicate truth table.
Extraction fails closed when anchors change. No GPU correctness is implied.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile


def between(text, start, end):
    assert text.count(start) == 1, f"production extraction anchor changed: {start}"
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


SCAFFOLD = r'''
#include "resident_batch_coordinator.h"
#include "batch_wake.h"
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <memory>
#include <stdexcept>
using namespace dflash::common;
using namespace std::chrono_literals;
std::atomic<long> pumps{0};
std::atomic<bool> throw_pump{false};
std::mutex hook_mu; std::condition_variable hook_cv;
bool hold_first=false, reached=false, resume_pump=false;
struct InstrumentedCoordinator : ResidentBatchCoordinator {
 using ResidentBatchCoordinator::ResidentBatchCoordinator;
 ContinuousBatchRunResult pump(int64_t t) {
  if(throw_pump.exchange(false))throw std::runtime_error("injected pump failure");
  auto r=ResidentBatchCoordinator::pump(t);
  if(++pumps==1 && hold_first){
   std::unique_lock<std::mutex> l(hook_mu);reached=true;hook_cv.notify_all();
   hook_cv.wait(l,[]{return resume_pump;});
  }
  return r;
 }
};
struct DummyModel {bool snapshot_used(int){return false;}int snapshot_cur_pos(int){return 0;}void shutdown(){}};
'''

BRIDGE = r'''
struct ember_backend {
 std::unique_ptr<DummyModel> be=std::make_unique<DummyModel>();std::unique_ptr<int> disk;
 ResidentBatchBackend *resident;
 int batch_sessions=2,batch_prefill_quantum=4,batch_mixed_prefill_quantum=2;
 int64_t batch_decode_coalesce_us=0;
 std::unique_ptr<InstrumentedCoordinator> coordinator;
 std::mutex batch_mu;std::condition_variable batch_cv;std::thread batch_thread;
 std::deque<ember_batch_call*> batch_pending;std::deque<ember_batch_control*> batch_controls;
 std::unordered_map<ContinuousBatchSessionId,ember_batch_call*> batch_active;
 bool batch_stop=false,batch_running=false,batch_start_done=false;std::string batch_start_error;
};
static int64_t batch_now_us(){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static void batch_refresh_stats_locked(ember_backend*){}
'''

TEST = r'''
void stop(ember_backend& b){
 {std::lock_guard<std::mutex> l(b.batch_mu);b.batch_stop=true;b.batch_cv.notify_all();}
 b.batch_thread.join();
}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::string mode=argv[1];FakeResidentBackend f;ember_backend b;b.resident=&f;
 hold_first=(mode=="control" || mode=="generation");
 b.batch_thread=std::thread(ember_batch_thread_main,&b);
 if(hold_first){std::unique_lock<std::mutex> l(hook_mu);hook_cv.wait(l,[]{return reached;});}
 else {std::unique_lock<std::mutex> l(b.batch_mu);b.batch_cv.wait(l,[&]{return b.batch_start_done;});}
 ember_batch_call call;call.request.prompt={7,7};call.request.n_gen=2;
 ember_batch_control control;control.fn=[]{};
 if(mode=="capacity"){
  control.fn=[&]{GenerateRequest q;q.n_gen=0;std::string e;
   for(int i=0;i<2;++i)if(!b.coordinator->admit(q,{},-1,0,&e))std::abort();};
  {std::unique_lock<std::mutex> l(b.batch_mu);b.batch_controls.push_back(&control);
   b.batch_cv.notify_one();control.cv.wait(l,[&]{return control.done;});}
  {std::lock_guard<std::mutex> l(b.batch_mu);b.batch_pending.push_back(&call);b.batch_cv.notify_one();}
  auto before=pumps.load();std::this_thread::sleep_for(50ms);auto count=pumps.load()-before;
  // A queued release must run even though terminal leases fill all slots.
  bool released=ember_batch_control_run(&b,[&]{auto ids=b.coordinator->sessions();
    if(!b.coordinator->release(ids.front()))std::abort();});
  bool done;
  {std::unique_lock<std::mutex> l(b.batch_mu);done=call.cv.wait_for(l,500ms,[&]{return call.done;});}
  stop(b);std::printf("capacity pumps=%ld release=%d generation=%d\n",count,released,done);
  return count>10 || !released || !done;
 }
 if(mode=="failure"){
  {std::unique_lock<std::mutex> l(b.batch_mu);throw_pump=true;
   b.batch_pending.push_back(&call);b.batch_cv.notify_one();call.cv.wait(l,[&]{return call.done;});}
  b.batch_thread.join();
  bool ran=false;bool serviced=ember_batch_control_run(&b,[&]{ran=true;});
  std::printf("failed request done=%d session=%llu running=%d control serviced=%d ran=%d\n",
   call.done,(unsigned long long)call.session_id,b.batch_running,serviced,ran);
  return !call.done || call.result.ok() || b.batch_running || serviced || ran;
 }
 {std::lock_guard<std::mutex> l(b.batch_mu);
  if(mode=="control")b.batch_controls.push_back(&control);else b.batch_pending.push_back(&call);
  b.batch_cv.notify_one();}
 {std::lock_guard<std::mutex> l(hook_mu);resume_pump=true;hook_cv.notify_all();}
 bool done;
 {std::unique_lock<std::mutex> l(b.batch_mu);
  if(mode=="control")done=control.cv.wait_for(l,500ms,[&]{return control.done;});
  else done=call.cv.wait_for(l,500ms,[&]{return call.done;});}
 stop(b);std::printf("%s completed before rescue=%d\n",mode.c_str(),done);
 return !done;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--flags", default="")
    args = parser.parse_args()
    root = args.source.resolve()
    bridge = (root / "src/backend/backend_dflash.cc").read_text()
    fake = between((root / "test/test_resident_batch_coordinator.cpp").read_text(),
                   "struct FakeResidentBackend", "\nstatic GenerateRequest request")
    structs = between(bridge, "struct ember_batch_call {", "\nstruct ember_backend {")
    fail = between(bridge, "static void batch_fail_call", "// batch_mu must be held.")
    discard = between(bridge, "static void ember_batch_discard_controls_locked",
                      "static void ember_batch_thread_main")
    worker = between(bridge, "static void ember_batch_thread_main", "static bool ember_batch_start")
    control = between(bridge, "template <typename Fn>\nstatic bool ember_batch_control_run",
                      'extern "C" ember_backend *ember_backend_load')
    code = (SCAFFOLD + fake + structs + BRIDGE + fail + discard +
            "\n#define ResidentBatchCoordinator InstrumentedCoordinator\n" + worker +
            "\n#undef ResidentBatchCoordinator\n" + control + TEST)
    common = root / "engine/dflash/common"
    with tempfile.TemporaryDirectory(prefix="ember-batch-worker-") as folder:
        cpp = Path(folder) / "worker.cpp"
        binary = Path(folder) / "worker"
        cpp.write_text(code)
        subprocess.run([args.compiler, *shlex.split(args.flags), "-std=c++17", "-pthread",
                        "-I" + str(common), "-I" + str(root / "engine/ggml/include"),
                        "-I" + str(root / "src/backend"), str(cpp),
                        *(str(common / (name + ".cpp")) for name in
                          ("resident_batch_coordinator", "continuous_batch_scheduler",
                           "continuous_batch_executor")), "-o", str(binary)],
                       check=True, timeout=60)
        for mode in ("control", "generation", "capacity", "failure"):
            # This failed once in a full ctest run straight after a parallel
            # build and passed on rerun and in 20 further runs. The CAUSE IS
            # NOT ESTABLISHED: ctest had already overwritten the log, so
            # whether it was this timeout or a scenario returning 1 is unknown,
            # and a passing rerun does not settle it. 30s is headroom for a
            # loaded runner, not a diagnosis -- raising it cannot hide a
            # scenario failure, which still exits non-zero.
            #
            # So report which happened. A recurrence has to be identifiable:
            # a timeout here looks exactly like the deadlock this test exists
            # to catch, and guessing between them is how a real regression gets
            # dismissed as flake.
            try:
                subprocess.run([str(binary), mode], check=True, timeout=30,
                               capture_output=True, text=True)
            except subprocess.TimeoutExpired as exc:
                raise SystemExit(
                    f"batch worker scenario {mode!r} TIMED OUT after 30s. "
                    f"Either the worker deadlocked -- which is the regression "
                    f"this test detects -- or the runner is badly overloaded. "
                    f"Do not dismiss this as flake without the distinction.\n"
                    f"stdout: {exc.stdout!r}") from exc
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    f"batch worker scenario {mode!r} FAILED (exit "
                    f"{exc.returncode}); this is a scenario assertion, not a "
                    f"timeout.\nstdout: {exc.stdout}\nstderr: {exc.stderr}"
                ) from exc


if __name__ == "__main__":
    main()
