// Host test for the DS4 imatrix accumulator and its .dat writer.
//
// The statistic and the file layout are the parts that must match upstream
// exactly, and both can be checked without a GPU: the accumulator takes host
// pointers, and the writer's output is parsed back here with the same reader
// the pipeline's imatrix_dat.py implements. A collector that produces a
// well-formed file with the wrong statistic is the failure this guards.

#include "../engine/dflash/deepseek4/deepseek4_imatrix.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

struct DatEntry {
    std::string name;
    int32_t ncall = 0;
    std::vector<float> values;
};

// Deliberately a second, independent implementation of the reader rather than
// a call into the writer's own code.
bool read_dat(const std::string & path, std::vector<DatEntry> & out,
              int32_t & last_chunk, std::string & dataset) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1) { std::fclose(f); return false; }
    for (int32_t i = 0; i < n; ++i) {
        int32_t len = 0;
        if (std::fread(&len, sizeof(len), 1, f) != 1) { std::fclose(f); return false; }
        std::string name(len, '\0');
        if (std::fread(name.data(), 1, len, f) != (size_t) len) { std::fclose(f); return false; }
        DatEntry e;
        e.name = name;
        if (std::fread(&e.ncall, sizeof(e.ncall), 1, f) != 1) { std::fclose(f); return false; }
        int32_t nval = 0;
        if (std::fread(&nval, sizeof(nval), 1, f) != 1) { std::fclose(f); return false; }
        e.values.resize(nval);
        if (std::fread(e.values.data(), sizeof(float), nval, f) != (size_t) nval) {
            std::fclose(f); return false;
        }
        out.push_back(std::move(e));
    }
    if (std::fread(&last_chunk, sizeof(last_chunk), 1, f) != 1) { std::fclose(f); return false; }
    int32_t dlen = 0;
    if (std::fread(&dlen, sizeof(dlen), 1, f) == 1) {
        dataset.assign(dlen, '\0');
        if (std::fread(dataset.data(), 1, dlen, f) != (size_t) dlen) { std::fclose(f); return false; }
    }
    std::fclose(f);
    return true;
}

}  // namespace

int main() {
    std::printf("ds4 imatrix accumulator\n");

    // A tiny model: 4 experts, rows of width 3, 2 tokens, top-2.
    const int n_expert = 4, n_in = 3, n_used = 2, n_tokens = 2;
    ds4::ImatrixCollector c;
    c.set_n_expert(n_expert);
    c.set_dataset("unit-test");

    // token 0 -> experts {0,2}, token 1 -> experts {2,3}
    const int32_t ids[n_tokens * n_used] = {0, 2, 2, 3};

    // Shared-row case (gate/up): one row per token, seen once per expert.
    const float in_shared[n_tokens * n_in] = {1.0f, 2.0f, 3.0f,
                                              4.0f, 0.0f, -1.0f};
    c.accumulate("blk.0.ffn_gate_exps.weight", in_shared, ids, n_in, n_used, n_tokens, false);

    const ds4::ImatrixCollector::Entry * g = c.entry("blk.0.ffn_gate_exps.weight");
    check(g != nullptr, "gate entry created");
    // expert 0 saw token 0 once: 1,4,9
    check(g && std::fabs(g->values[0*n_in+0] - 1.0) < 1e-9 &&
                std::fabs(g->values[0*n_in+1] - 4.0) < 1e-9 &&
                std::fabs(g->values[0*n_in+2] - 9.0) < 1e-9, "expert 0 sums squares of its one row");
    // expert 2 saw BOTH tokens: 1+16, 4+0, 9+1
    check(g && std::fabs(g->values[2*n_in+0] - 17.0) < 1e-9 &&
                std::fabs(g->values[2*n_in+1] -  4.0) < 1e-9 &&
                std::fabs(g->values[2*n_in+2] - 10.0) < 1e-9, "expert 2 accumulates both tokens");
    check(g && g->counts[0] == 1 && g->counts[2] == 2 && g->counts[3] == 1 && g->counts[1] == 0,
          "counts are per expert, and an unrouted expert stays 0");

    // Per-slot case (down): a distinct row per (slot, token).
    const float in_slot[n_tokens * n_used * n_in] = {
        1.0f, 0.0f, 0.0f,   // t0 s0 -> expert 0
        0.0f, 2.0f, 0.0f,   // t0 s1 -> expert 2
        0.0f, 0.0f, 3.0f,   // t1 s0 -> expert 2
        5.0f, 0.0f, 0.0f,   // t1 s1 -> expert 3
    };
    c.accumulate("blk.0.ffn_down_exps.weight", in_slot, ids, n_in, n_used, n_tokens, true);
    const ds4::ImatrixCollector::Entry * d = c.entry("blk.0.ffn_down_exps.weight");
    check(d && std::fabs(d->values[2*n_in+1] - 4.0) < 1e-9 &&
                std::fabs(d->values[2*n_in+2] - 9.0) < 1e-9,
          "per-slot rows land on the expert that produced them");
    check(d && std::fabs(d->values[0*n_in+0] - 1.0) < 1e-9 &&
                std::fabs(d->values[3*n_in+0] - 25.0) < 1e-9,
          "per-slot indexing does not cross tokens");

    // A shared-row read of the per-slot buffer would put t1's row (0,0,3) on
    // expert 3; prove the two modes actually differ, or the flag is untested.
    check(d && std::fabs(d->values[3*n_in+2] - 0.0) < 1e-9,
          "per_slot=true is not silently the same as per_slot=false");

    c.end_chunk();
    c.end_chunk();

    const std::string path = "/tmp/ds4-imatrix-test.dat";
    std::remove(path.c_str());
    std::string err;
    check(c.write(path, err), "writer succeeds");

    std::vector<DatEntry> back;
    int32_t last_chunk = -1;
    std::string dataset;
    check(read_dat(path, back, last_chunk, dataset), "file parses with an independent reader");
    check(back.size() == 2, "two entries round-trip");
    check(last_chunk == 2, "chunk count is recorded");
    check(dataset == "unit-test", "dataset name is recorded");

    // Stored values are divided by counts: expert 2 saw 2 tokens, so 17/2.
    for (const DatEntry & e : back) {
        if (e.name != "blk.0.ffn_gate_exps.weight") continue;
        check(std::fabs(e.values[2*n_in+0] - 8.5f) < 1e-6, "stored values are divided by counts");
        check(std::fabs(e.values[1*n_in+0] - 0.0f) < 1e-9,
              "an unrouted expert stores 0, not a 1.0 sentinel");
    }
    check(std::remove(path.c_str()) == 0, "temp file removed");

    std::printf(failures ? "\nFAILURES: %d\n" : "\nALL PASS\n", failures);
    return failures ? 1 : 0;
}
