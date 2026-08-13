#include "deepseek4_image_embed.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

constexpr char   MAGIC[8]     = {'D', 'S', '4', 'I', 'M', 'G', 'E', '1'};
constexpr size_t MAX_TOKENS   = 4096;      // sanity bound; upstream trains to 512
constexpr size_t MAX_PALETTE  = 4096;

[[noreturn]] void fail(const char * what) {
    std::fprintf(stderr,
                 "[ds4-image] FATAL: %s\n"
                 "[ds4-image] EMBER_DS4_IMAGE_EMBED is set, so the run was meant to "
                 "carry an image; refusing to serve text-only and pretend it worked.\n",
                 what);
    std::exit(1);
}

bool read_exact(std::FILE * f, void * dst, size_t n) {
    return std::fread(dst, 1, n, f) == n;
}

static uint64_t ds4_image_digest(const Ds4ImageEmbed & e) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void * p, size_t n) {
        const unsigned char * b = (const unsigned char *) p;
        for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    mix(&e.n_tokens, sizeof(e.n_tokens));
    mix(&e.n_embd,   sizeof(e.n_embd));
    if (!e.palette.empty()) mix(e.palette.data(), e.palette.size() * sizeof(int32_t));
    if (!e.data.empty())    mix(e.data.data(),    e.data.size()    * sizeof(float));
    return h ? h : 1ULL;
}

Ds4ImageEmbed load(int64_t model_n_embd) {
    Ds4ImageEmbed e;
    const char * path = std::getenv("EMBER_DS4_IMAGE_EMBED");
    if (!path || !*path) return e;   // inactive: the ordinary text path

    std::FILE * f = std::fopen(path, "rb");
    if (!f) fail("cannot open $EMBER_DS4_IMAGE_EMBED");

    char magic[8];
    int32_t hdr[4];
    if (!read_exact(f, magic, sizeof(magic)) || std::memcmp(magic, MAGIC, 8) != 0)
        fail("bad magic (expected DS4IMGE1)");
    if (!read_exact(f, hdr, sizeof(hdr))) fail("truncated header");

    e.n_embd        = hdr[0];
    e.n_tokens      = hdr[1];
    const int32_t np = hdr[2];

    if (e.n_embd != (int32_t) model_n_embd) {
        std::fprintf(stderr, "[ds4-image] sidecar n_embd %d != model n_embd %lld\n",
                     e.n_embd, (long long) model_n_embd);
        fail("embedding width mismatch");
    }
    if (e.n_tokens <= 0 || (size_t) e.n_tokens > MAX_TOKENS) fail("implausible n_tokens");
    if (np <= 0 || (size_t) np > MAX_PALETTE)                fail("implausible n_palette");

    e.palette.resize((size_t) np);
    if (!read_exact(f, e.palette.data(), sizeof(int32_t) * (size_t) np))
        fail("truncated palette");

    e.data.resize((size_t) e.n_tokens * (size_t) e.n_embd);
    if (!read_exact(f, e.data.data(), sizeof(float) * e.data.size()))
        fail("truncated embedding data");

    // A trailing byte means the writer and this reader disagree about the
    // format, which is exactly the kind of drift that produces silent garbage.
    char extra;
    if (std::fread(&extra, 1, 1, f) != 0) fail("trailing bytes after embedding data");
    std::fclose(f);

    e.active = true;

    // Digest the SPLICED CONTENT -- the embedding rows and the palette that
    // addresses them -- not the file path or mtime. Two sidecars written from
    // the same picture must collide (so the cache still helps on a repeat), and
    // two different pictures must not (so image B can never restore image A's
    // KV behind identical palette tokens).
    e.digest = ds4_image_digest(e);

    std::fprintf(stderr,
                 "[ds4-image] loaded %s: %d image tokens x %d, palette %d, "
                 "digest %016llx\n",
                 path, e.n_tokens, e.n_embd, np,
                 (unsigned long long) e.digest);
    return e;
}

// FNV-1a over the embedding bytes plus the palette and shape. Not a security
// hash -- it only has to separate distinct images inside one process, where the
// alternative (token IDs alone) cannot separate them at all. Forced non-zero so
// 0 stays an unambiguous "no image loaded".
Ds4ImageEmbed   g_embed;
std::once_flag  g_once;

}  // namespace

const Ds4ImageEmbed & Ds4ImageEmbed::instance(int64_t model_n_embd) {
    std::call_once(g_once, [model_n_embd] { g_embed = load(model_n_embd); });
    return g_embed;
}

const Ds4ImageEmbed & Ds4ImageEmbed::loaded() { return g_embed; }

int64_t Ds4ImageEmbed::find_span(const std::vector<int32_t> & tokens,
                                 std::string * err) const {
    if (!active || palette.empty()) return -1;
    const size_t n = tokens.size();
    const size_t want = (size_t) n_tokens;
    if (n < want) return -1;

    for (size_t s = 0; s + 1 <= n; ++s) {
        if (tokens[s] != palette[0]) continue;
        size_t k = 1;
        while (k < want && s + k < n && tokens[s + k] == palette[k % palette.size()]) ++k;
        if (k == want) return (int64_t) s;
        if (k >= palette.size()) {
            // Matched at least a full cycle then stopped: this is the image, but
            // the prompt carries a different number of positions than the
            // sidecar has embeddings. Never splice through that.
            if (err) {
                char buf[192];
                std::snprintf(buf, sizeof(buf),
                              "image span at %zu is %zu tokens but the sidecar holds %zu; "
                              "prompt and embeddings disagree", s, k, want);
                *err = buf;
            }
            return -1;
        }
    }
    return -1;
}

void Ds4ImageEmbed::splice(float * embed, int64_t chunk_start, int n_tok,
                           int64_t span_start) const {
    if (!active || span_start < 0) return;
    for (int t = 0; t < n_tok; ++t) {
        const int64_t p = chunk_start + t;
        if (p < span_start || p >= span_start + n_tokens) continue;
        std::memcpy(embed + (size_t) t * (size_t) n_embd,
                    data.data() + (size_t) (p - span_start) * (size_t) n_embd,
                    sizeof(float) * (size_t) n_embd);
    }
}

extern "C" uint64_t ember_ds4_image_digest(void) {
    return g_embed.active ? g_embed.digest : 0;
}

extern "C" int ember_ds4_image_span_start(const int32_t * ids, int n) {
    if (!g_embed.active || !ids || n <= 0) return -1;
    std::vector<int32_t> v(ids, ids + n);
    std::string err;
    const int64_t s = g_embed.find_span(v, &err);
    // A partial match means prompt and sidecar disagree about the image length.
    // The splice path treats that as fatal. Here the safe reading is "no usable
    // span", which makes the cache refuse to tag the entry rather than tag it
    // at a position the backend will reject anyway.
    if (s < 0) return -1;
    return (int) s;
}

extern "C" int ember_ds4_image_token_count(const int32_t ** palette_out,
                                           int * n_palette_out) {
    // n_embd is validated on the first real load inside the backend; passing 0
    // here would trip the width check, so this shim only reports a sidecar that
    // the backend has already accepted.
    if (!g_embed.active) return 0;
    if (palette_out)   *palette_out   = g_embed.palette.data();
    if (n_palette_out) *n_palette_out = (int) g_embed.palette.size();
    return g_embed.n_tokens;
}
