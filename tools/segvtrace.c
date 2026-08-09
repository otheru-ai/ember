// Minimal crash-backtrace shim for dflash_server.
// A real core is impractical here (~103 GB anonymous RSS), and systemd-coredump
// skips the process on ProcessSizeMax. This prints a symbolized backtrace to
// stderr on fatal signals, which systemd captures in the unit journal.
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <dlfcn.h>
#include <stdint.h>
#include <sys/syscall.h>

static struct sigaction old_segv, old_bus, old_fpe, old_ill, old_abrt;

// ── crash breadcrumb ────────────────────────────────────────────────────────
// The engine keeps the last known hot-path state in a global (see
// engine/dflash/common/crash_breadcrumb.h). A backtrace gives the call path but
// no state, which is not enough to explain a SIGSEGV inside an indexed load.
// Resolve it lazily by name so this shim still works against a binary that does
// not export it (older build, or a non-DeepSeek4 server).
struct ds4_crash_bc {
    volatile int32_t site, layer_idx, kv_start, n_tokens, ratio, ratios_size;
    volatile int32_t impl, n_layer, cache_pos, spec_pos, spec_q, spec_accept;
    volatile int32_t spec_ngen, restore_snap, restore_prompt, committed;
    volatile int64_t seq;
};

struct ds4_kv_view_bc {
    volatile uint32_t version, size;
    volatile int32_t layer_idx, row, type, reserved;
    volatile uint64_t tensor, view_src, ne0, ne1, nb0, nb1;
    volatile uint64_t view_offs, relative_offs, absolute_offs;
    volatile uint64_t requested_bytes, base_nbytes;
    volatile uint64_t created_view_offs, created_base_nbytes, seq;
};

static const char *bc_site_name(int s) {
    switch (s) {
        case 1: return "build_mla_attention";
        case 2: return "deepseek4_step_layer_range";
        case 3: return "dspark_verify_forward";
        case 4: return "dspark_spec_decode";
        case 5: return "restore_and_generate_impl";
        default: return "none";
    }
}

static void dump_breadcrumb(void) {
    const struct ds4_crash_bc *b =
        (const struct ds4_crash_bc *) dlsym(RTLD_DEFAULT, "ds4_crash_breadcrumb");
    if (!b) {
        const char *m = "[segvtrace] no ds4_crash_breadcrumb symbol (uninstrumented binary)\n";
        ssize_t w = write(2, m, strlen(m)); (void)w;
        return;
    }
    char buf[512];
    int n = snprintf(buf, sizeof buf,
        "=== [segvtrace] breadcrumb seq=%lld site=%d(%s) ===\n"
        "  mla:     layer_idx=%d n_layer=%d ratios_size=%d ratio=%d%s\n"
        "  mla:     kv_start=%d n_tokens=%d impl=%d lc.n_comp=%d\n"
        "  spec:    pos=%d q=%d accept=%d ngen=%d\n"
        "  restore: snap_pos=%d prompt_len=%d committed=%d\n",
        (long long) b->seq, b->site, bc_site_name(b->site),
        b->layer_idx, b->n_layer, b->ratios_size, b->ratio,
        (b->ratios_size > 0 && (b->layer_idx < 0 || b->layer_idx >= b->ratios_size))
            ? "  <-- layer_idx OUT OF BOUNDS" : "",
        b->kv_start, b->n_tokens, b->impl, b->cache_pos,
        b->spec_pos, b->spec_q, b->spec_accept, b->spec_ngen,
        b->restore_snap, b->restore_prompt, b->committed);
    ssize_t w = write(2, buf, n > 0 ? (size_t) n : 0); (void)w;

    const struct ds4_kv_view_bc *kv =
        (const struct ds4_kv_view_bc *)
            dlsym(RTLD_DEFAULT, "ds4_kv_view_breadcrumb");
    if (!kv || kv->version != 1 || kv->size < sizeof(*kv)) return;
    n = snprintf(buf, sizeof buf,
        "  raw-kv:  seq=%llu layer=%d row=%d type=%d tensor=%p view_src=%p\n"
        "  raw-kv:  ne=[%llu,%llu] nb=[%llu,%llu] view_offs=%llu "
        "relative=%llu absolute=%llu requested=%llu base_nbytes=%llu\n"
        "  raw-kv:  created_view_offs=%llu created_base_nbytes=%llu\n",
        (unsigned long long) kv->seq, kv->layer_idx, kv->row, kv->type,
        (void *)(uintptr_t) kv->tensor, (void *)(uintptr_t) kv->view_src,
        (unsigned long long) kv->ne0, (unsigned long long) kv->ne1,
        (unsigned long long) kv->nb0, (unsigned long long) kv->nb1,
        (unsigned long long) kv->view_offs,
        (unsigned long long) kv->relative_offs,
        (unsigned long long) kv->absolute_offs,
        (unsigned long long) kv->requested_bytes,
        (unsigned long long) kv->base_nbytes,
        (unsigned long long) kv->created_view_offs,
        (unsigned long long) kv->created_base_nbytes);
    w = write(2, buf, n > 0 ? (size_t) n : 0); (void)w;
}

static void handler(int sig, siginfo_t *si, void *uc) {
    static volatile sig_atomic_t in_handler = 0;
    if (in_handler) _exit(128 + sig);   // avoid recursion
    in_handler = 1;

    char buf[256];
    int n = snprintf(buf, sizeof buf,
        "\n=== [segvtrace] FATAL signal %d (%s) at addr %p, pid %d, tid %ld ===\n",
        sig, strsignal(sig), si ? si->si_addr : NULL,
        (int)getpid(), (long)gettid());
    ssize_t w = write(2, buf, n); (void)w;

    void *frames[64];
    int nf = backtrace(frames, 64);
    backtrace_symbols_fd(frames, nf, 2);

    n = snprintf(buf, sizeof buf, "=== [segvtrace] end backtrace (%d frames) ===\n", nf);
    w = write(2, buf, n); (void)w;

    // State the backtrace cannot give us: which layer/index was live.
    dump_breadcrumb();

    // restore default and re-raise so the exit status still reflects the signal
    struct sigaction dfl; memset(&dfl, 0, sizeof dfl); dfl.sa_handler = SIG_DFL;
    sigaction(sig, &dfl, NULL);
    raise(sig);
}

__attribute__((constructor))
static void install(void) {
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS,  &sa, &old_bus);
    sigaction(SIGFPE,  &sa, &old_fpe);
    sigaction(SIGILL,  &sa, &old_ill);
    sigaction(SIGABRT, &sa, &old_abrt);
    const char *m = "[segvtrace] fatal-signal backtrace handler installed\n";
    ssize_t w = write(2, m, strlen(m)); (void)w;
}
