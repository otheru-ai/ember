#pragma once

// Importance-matrix collection for the DeepSeek-V4 routed experts.
//
// Why this lives in ember rather than in a fork of llama.cpp's llama-imatrix:
// an imatrix is only valid for the routing it was collected through. Upstream's
// collector cannot feed images to this tower at all -- tools/imatrix/imatrix.cpp
// references neither mtmd nor clip, and no PR proposes it -- and the one public
// vision fork routes image tokens correctly but falls back to a causal window
// instead of the reference's in-span bidirectional visibility, which perturbs
// every activation the matrix measures. ember already runs images end to end
// with `exp_probs_b_vl` selection (deepseek4_graph.cpp:4736) and in-span
// visibility (deepseek4_vision_contract.cpp:444), at the serving top-k. So the
// calibration is collected here, through the exact path that will serve it.
//
// The statistic is upstream's, unchanged, so the output stays consumable by the
// archived affine writer: for a GGML_OP_MUL_MAT_ID node, for each token routed
// to expert `ex`, accumulate `values[ex*n_in + j] += x[j]^2` over that token's
// input row and increment `counts[ex]`. The legacy .dat stores values already
// divided by counts.
//
// ember does not use ggml_backend_sched, so there is no eval callback to hang
// this on (`ggml_backend_graph_compute` is called directly). Instead the graph
// builder REGISTERS the input rows and the routing ids while the graph is being
// built -- marking them ggml_set_output so gallocr does not reuse their buffers
// -- and the caller DRAINS them after the compute.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ggml_tensor;

namespace ds4 {

// One registered MUL_MAT_ID site for one layer of one graph.
struct ImatrixSite {
    std::string   name;      // GGUF tensor name, e.g. "blk.7.ffn_gate_exps.weight"
    ggml_tensor * input;     // rows fed to the experts
    ggml_tensor * ids;       // [n_used, n_tokens] i32 expert selection
    int           n_in;      // row width; nval is n_in * n_expert
    bool          per_slot;  // false: one row per token (gate/up)
                             // true:  one row per (slot, token) (down)
};

class ImatrixCollector {
public:
    // Returns nullptr unless DFLASH_IMATRIX_OUT is set. One instance per
    // process; collection is off in every normal serving path.
    static ImatrixCollector * instance();

    // Registration is ignored outside a scope. Several graph builders reach the
    // MoE code, but only one of them is followed by a drain; without this gate
    // the others leave sites_ holding pointers into a freed ggml_context for
    // the next drain to dereference. Scope is opened only where a drain is
    // guaranteed.
    void begin_scope();
    void end_scope();
    bool in_scope() const { return scope_; }

    // Called from the graph builder. Marks `input` and `ids` as graph outputs.
    // No-op outside a scope.
    void register_site(const std::string & name, ggml_tensor * input,
                       ggml_tensor * ids, int n_in, bool per_slot);

    // Called after ggml_backend_graph_compute. Reads every registered site back
    // to host, accumulates, and clears the registry for the next graph.
    void drain();

    // Discard registrations without reading them: a graph that failed to
    // compute holds undefined buffers, and a short record an extractor mistakes
    // for evidence is exactly the failure this design is meant to avoid.
    void abandon();

    // Counts one calibration chunk (one request/graph sequence).
    void end_chunk();

    // Writes the legacy .dat the archived affine writer consumes. Atomic:
    // writes a temp file in the same directory and renames, so a killed process
    // never leaves a truncated matrix behind.
    bool write(const std::string & path, std::string & err) const;

    int  n_expert() const { return n_expert_; }
    int  chunks()   const { return chunks_; }
    void set_n_expert(int n) { n_expert_ = n; }
    void set_dataset(const std::string & d) { dataset_ = d; }

    // Accumulate one already-host-side site. Exposed for the host unit test,
    // which must be able to check the statistic without a GPU.
    void accumulate(const std::string & name, const float * input,
                    const int32_t * ids, int n_in, int n_used, int n_tokens,
                    bool per_slot);

    struct Entry {
        std::vector<double> values;  // n_in * n_expert, sum of squares
        std::vector<int64_t> counts; // n_expert
        int n_in = 0;
        int ncall = 0;
    };
    const Entry * entry(const std::string & name) const;

private:
    // The generation workers are separate threads and the collector is a
    // process-wide singleton, so every mutating path takes this. Collection is
    // off in normal serving, but a batched collection run would otherwise
    // corrupt counts rather than fail.
    mutable std::mutex mu_;
    bool scope_ = false;
    std::vector<ImatrixSite> sites_;
    // std::map keeps the .dat entry order stable across runs, which makes two
    // collections byte-comparable.
    std::vector<std::pair<std::string, Entry>> entries_;
    Entry * entry_mut(const std::string & name, int n_in);
    int n_expert_ = 256;
    int chunks_ = 0;
    std::string dataset_;
};

}  // namespace ds4
