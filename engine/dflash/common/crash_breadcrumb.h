#pragma once
// Crash breadcrumb: the last known state on the DeepSeek4 hot path.
//
// A core dump is impractical for this process (~93 GB of anonymous, unevictable
// RSS — systemd-coredump skips it on ProcessSizeMax), so tools/segvtrace.c
// prints a symbolized backtrace on a fatal signal instead. A backtrace alone
// gives the call path but none of the state, which is not enough to explain a
// SIGSEGV in build_mla_attention: we need the indices that were live.
//
// This is that state. Writes are plain scalar stores into a single global (a
// few nanoseconds, no formatting, no locking), so it is cheap enough to leave
// permanently enabled on the hot path — which matters because the crash it was
// added for is rare and only reproduces in production. Formatting happens once,
// inside the signal handler, after the process is already dying.
//
// segvtrace resolves the symbol with dlsym(RTLD_DEFAULT, "ds4_crash_breadcrumb")
// and prints it, so it degrades to today's behaviour if the symbol is absent
// (older binary + newer shim, or vice versa).
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Site identifiers. Keep the numbering stable — segvtrace prints the name.
enum {
    DS4_BC_NONE            = 0,
    DS4_BC_MLA_ATTENTION   = 1,  // build_mla_attention entry
    DS4_BC_STEP_LAYERS     = 2,  // deepseek4_step_layer_range entry
    DS4_BC_SPEC_VERIFY     = 3,  // dspark verify_forward entry
    DS4_BC_SPEC_STEP       = 4,  // dspark spec-decode per-block state
    DS4_BC_RESTORE_GEN     = 5,  // restore_and_generate_impl entry
};

// Single global, volatile so the compiler keeps the stores in program order
// rather than sinking them past the faulting instruction.
struct ds4_crash_bc {
    volatile int32_t site;        // one of DS4_BC_*
    volatile int32_t layer_idx;   // MLA: layer being built
    volatile int32_t kv_start;    // MLA: KV base position
    volatile int32_t n_tokens;    // MLA/verify: tokens in this pass
    volatile int32_t ratio;       // MLA: compress_ratios[layer_idx] (-1 = not yet read)
    volatile int32_t ratios_size; // MLA: compress_ratios.size() — layer_idx >= this is OOB
    volatile int32_t impl;        // MLA: DeepSeek4AttentionImpl
    volatile int32_t n_layer;     // weights layer count (bounds check vs layer_idx)
    volatile int32_t cache_pos;   // MLA: layer cache n_comp at entry
    volatile int32_t spec_pos;    // spec: current position
    volatile int32_t spec_q;      // spec: verify width this block
    volatile int32_t spec_accept; // spec: accepted count last block
    volatile int32_t spec_ngen;   // spec: tokens emitted so far / budget
    volatile int32_t restore_snap;// restore: snapshot position
    volatile int32_t restore_prompt; // restore: prompt length
    volatile int32_t committed;   // committed KV position
    volatile int64_t seq;         // monotonic write counter (detects staleness)
};

extern struct ds4_crash_bc ds4_crash_breadcrumb;

// Separate versioned symbol for the raw-KV view invariant. Keeping this out of
// ds4_crash_bc preserves that symbol's ABI when a newer LD_PRELOAD shim is used
// with an older server binary.
struct ds4_kv_view_bc {
    volatile uint32_t version;
    volatile uint32_t size;
    volatile int32_t layer_idx;
    volatile int32_t row;
    volatile int32_t type;
    volatile int32_t reserved;
    volatile uint64_t tensor;
    volatile uint64_t view_src;
    volatile uint64_t ne0;
    volatile uint64_t ne1;
    volatile uint64_t nb0;
    volatile uint64_t nb1;
    volatile uint64_t view_offs;
    volatile uint64_t relative_offs;
    volatile uint64_t absolute_offs;
    volatile uint64_t requested_bytes;
    volatile uint64_t base_nbytes;
    volatile uint64_t created_view_offs;
    volatile uint64_t created_base_nbytes;
    volatile uint64_t seq;
};

extern struct ds4_kv_view_bc ds4_kv_view_breadcrumb;

// Hot-path setters. Deliberately plain stores; no snprintf, no locks.
// Recorded at build_mla_attention entry, BEFORE compress_ratios[layer_idx] is
// read — that indexed load is itself a crash candidate, so the breadcrumb must
// already hold layer_idx and the array bound when it faults. ratio is filled in
// afterwards via ds4_bc_mla_ratio().
static inline void ds4_bc_mla(int layer_idx, int kv_start, int n_tokens,
                              int ratios_size, int impl, int n_layer, int cache_pos) {
    ds4_crash_breadcrumb.site        = DS4_BC_MLA_ATTENTION;
    ds4_crash_breadcrumb.layer_idx   = layer_idx;
    ds4_crash_breadcrumb.kv_start    = kv_start;
    ds4_crash_breadcrumb.n_tokens    = n_tokens;
    ds4_crash_breadcrumb.ratio       = -1;
    ds4_crash_breadcrumb.ratios_size = ratios_size;
    ds4_crash_breadcrumb.impl        = impl;
    ds4_crash_breadcrumb.n_layer     = n_layer;
    ds4_crash_breadcrumb.cache_pos   = cache_pos;
    ds4_crash_breadcrumb.seq++;
}

static inline void ds4_bc_mla_ratio(int ratio) {
    ds4_crash_breadcrumb.ratio = ratio;
}

static inline void ds4_bc_kv_view(
        int layer_idx, int row, int type,
        uint64_t tensor, uint64_t view_src,
        uint64_t ne0, uint64_t ne1, uint64_t nb0, uint64_t nb1,
        uint64_t view_offs, uint64_t relative_offs, uint64_t absolute_offs,
        uint64_t requested_bytes, uint64_t base_nbytes,
        uint64_t created_view_offs, uint64_t created_base_nbytes) {
    ds4_kv_view_breadcrumb.version             = 1;
    ds4_kv_view_breadcrumb.size                = sizeof(ds4_kv_view_breadcrumb);
    ds4_kv_view_breadcrumb.layer_idx           = layer_idx;
    ds4_kv_view_breadcrumb.row                 = row;
    ds4_kv_view_breadcrumb.type                = type;
    ds4_kv_view_breadcrumb.tensor              = tensor;
    ds4_kv_view_breadcrumb.view_src            = view_src;
    ds4_kv_view_breadcrumb.ne0                 = ne0;
    ds4_kv_view_breadcrumb.ne1                 = ne1;
    ds4_kv_view_breadcrumb.nb0                 = nb0;
    ds4_kv_view_breadcrumb.nb1                 = nb1;
    ds4_kv_view_breadcrumb.view_offs           = view_offs;
    ds4_kv_view_breadcrumb.relative_offs       = relative_offs;
    ds4_kv_view_breadcrumb.absolute_offs       = absolute_offs;
    ds4_kv_view_breadcrumb.requested_bytes     = requested_bytes;
    ds4_kv_view_breadcrumb.base_nbytes         = base_nbytes;
    ds4_kv_view_breadcrumb.created_view_offs   = created_view_offs;
    ds4_kv_view_breadcrumb.created_base_nbytes = created_base_nbytes;
    ds4_kv_view_breadcrumb.seq++;
}

static inline void ds4_bc_site(int site) {
    ds4_crash_breadcrumb.site = site;
    ds4_crash_breadcrumb.seq++;
}

static inline void ds4_bc_spec(int pos, int q, int accept, int ngen, int committed) {
    ds4_crash_breadcrumb.site        = DS4_BC_SPEC_STEP;
    ds4_crash_breadcrumb.spec_pos    = pos;
    ds4_crash_breadcrumb.spec_q      = q;
    ds4_crash_breadcrumb.spec_accept = accept;
    ds4_crash_breadcrumb.spec_ngen   = ngen;
    ds4_crash_breadcrumb.committed   = committed;
    ds4_crash_breadcrumb.seq++;
}

static inline void ds4_bc_restore(int snap_pos, int prompt_len, int committed) {
    ds4_crash_breadcrumb.site           = DS4_BC_RESTORE_GEN;
    ds4_crash_breadcrumb.restore_snap   = snap_pos;
    ds4_crash_breadcrumb.restore_prompt = prompt_len;
    ds4_crash_breadcrumb.committed      = committed;
    ds4_crash_breadcrumb.seq++;
}

#ifdef __cplusplus
}
#endif
