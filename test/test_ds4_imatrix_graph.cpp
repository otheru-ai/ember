// End-to-end test of the imatrix collector against a REAL ggml graph.
//
// test_ds4_imatrix checks the statistic with host pointers. This checks the
// half that host pointers cannot: that registering a ggml_reshape VIEW as a
// graph output actually survives the graph allocator, that the byte counts
// drain() computes match what those tensors really hold, and that the ids
// layout is the one the accumulator assumes. Those are the assumptions that
// would silently produce a well-formed matrix full of the wrong numbers.
//
// Built on the CPU backend, so it runs anywhere; the graph mirrors the shapes
// the DS4 MoE builds: cur_3d = [n_embd, 1, n_tokens] shared rows for gate/up,
// mid_e = [n_ff, n_used, n_tokens] per-slot rows for down, ids = [n_used,
// n_tokens] i32.

#include "../engine/dflash/deepseek4/deepseek4_imatrix.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;
static void check(bool ok, const char * what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

int main() {
    std::printf("ds4 imatrix against a real ggml graph\n");

    const int n_embd = 8, n_ff = 4, n_expert = 6, n_used = 2, n_tokens = 3;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { std::printf("  no CPU backend\n"); return 1; }

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // Inputs, shaped as the DS4 graph shapes them.
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(cur);
    ggml_tensor * mid = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_used, n_tokens);
    ggml_set_input(mid);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(ids);

    // The registration site sees cur as a reshape VIEW, exactly as the engine
    // builds it. If ggml_set_output on a view does not protect the base, the
    // read below returns something other than what was written.
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

    // Real mul_mat_id nodes, so cur_3d/mid/ids are consumed exactly as the
    // engine consumes them. An earlier version of this test used a dummy add;
    // gallocr then never allocated the unconsumed inputs and the test aborted
    // in ggml_backend_tensor_set -- which is the right behaviour, and the
    // reason the graph here has to mirror production rather than approximate.
    ggml_tensor * w_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_set_input(w_gate);
    ggml_tensor * w_down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_expert);
    ggml_set_input(w_down);
    ggml_tensor * gate_e = ggml_mul_mat_id(ctx, w_gate, cur_3d, ids);
    ggml_tensor * down_e = ggml_mul_mat_id(ctx, w_down, mid, ids);
    ggml_tensor * consumed = ggml_add(ctx,
        ggml_reshape_1d(ctx, gate_e, ggml_nelements(gate_e)),
        ggml_reshape_1d(ctx, gate_e, ggml_nelements(gate_e)));
    ggml_set_output(consumed);
    ggml_set_output(down_e);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, consumed);
    ggml_build_forward_expand(gf, down_e);

    ds4::ImatrixCollector imx;
    imx.set_n_expert(n_expert);
    imx.register_site("blk.0.ffn_gate_exps.weight", cur_3d, ids, n_embd, false);
    imx.register_site("blk.0.ffn_down_exps.weight", mid,    ids, n_ff,   true);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    check(ggml_gallocr_alloc_graph(alloc, gf), "graph allocates with registered outputs");

    std::vector<float> h_cur((size_t) n_embd * n_tokens);
    for (size_t i = 0; i < h_cur.size(); ++i) h_cur[i] = (float) (i + 1);
    std::vector<float> h_mid((size_t) n_ff * n_used * n_tokens);
    for (size_t i = 0; i < h_mid.size(); ++i) h_mid[i] = (float) (i % 5) - 2.0f;
    // token 0 -> {0,3}, token 1 -> {3,5}, token 2 -> {1,3}
    const int32_t h_ids[n_used * n_tokens] = {0, 3, 3, 5, 1, 3};

    std::vector<float> h_w((size_t) n_embd * n_ff * n_expert, 0.25f);
    ggml_backend_tensor_set(w_gate, h_w.data(), 0, sizeof(float) * h_w.size());
    ggml_backend_tensor_set(w_down, h_w.data(), 0, sizeof(float) * h_w.size());
    ggml_backend_tensor_set(cur, h_cur.data(), 0, sizeof(float) * h_cur.size());
    ggml_backend_tensor_set(mid, h_mid.data(), 0, sizeof(float) * h_mid.size());
    ggml_backend_tensor_set(ids, h_ids, 0, sizeof(h_ids));

    check(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS, "graph computes");
    imx.drain();

    // Reference, computed here from the host arrays the graph was fed.
    std::vector<double> ref_gate((size_t) n_embd * n_expert, 0.0);
    std::vector<long>   ref_cnt(n_expert, 0);
    for (int t = 0; t < n_tokens; ++t) {
        for (int s = 0; s < n_used; ++s) {
            const int ex = h_ids[t * n_used + s];
            for (int j = 0; j < n_embd; ++j) {
                const double v = h_cur[(size_t) t * n_embd + j];
                ref_gate[(size_t) ex * n_embd + j] += v * v;
            }
            ref_cnt[ex]++;
        }
    }
    const ds4::ImatrixCollector::Entry * g = imx.entry("blk.0.ffn_gate_exps.weight");
    bool gate_ok = g != nullptr;
    if (g) {
        for (size_t i = 0; i < ref_gate.size() && gate_ok; ++i) {
            gate_ok = std::fabs(g->values[i] - ref_gate[i]) < 1e-6;
        }
        for (int e = 0; e < n_expert && gate_ok; ++e) gate_ok = g->counts[e] == ref_cnt[e];
    }
    check(gate_ok, "shared rows read back through a reshape view match the host reference");

    std::vector<double> ref_down((size_t) n_ff * n_expert, 0.0);
    for (int t = 0; t < n_tokens; ++t) {
        for (int s = 0; s < n_used; ++s) {
            const int ex = h_ids[t * n_used + s];
            for (int j = 0; j < n_ff; ++j) {
                const double v = h_mid[(((size_t) t * n_used) + s) * n_ff + j];
                ref_down[(size_t) ex * n_ff + j] += v * v;
            }
        }
    }
    const ds4::ImatrixCollector::Entry * d = imx.entry("blk.0.ffn_down_exps.weight");
    bool down_ok = d != nullptr;
    if (d) {
        for (size_t i = 0; i < ref_down.size() && down_ok; ++i) {
            down_ok = std::fabs(d->values[i] - ref_down[i]) < 1e-6;
        }
    }
    check(down_ok, "per-slot rows read back match the host reference");

    // The registry must be empty after a drain, or the next graph double counts.
    imx.drain();
    bool unchanged = true;
    if (g) {
        for (size_t i = 0; i < ref_gate.size() && unchanged; ++i) {
            unchanged = std::fabs(g->values[i] - ref_gate[i]) < 1e-6;
        }
    }
    check(unchanged, "a second drain with no registrations does not double count");

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    std::printf(failures ? "\nFAILURES: %d\n" : "\nALL PASS\n", failures);
    return failures ? 1 : 0;
}
