// Span-location tests for the vision graft's validation path.
//
// find_span() locates image positions by matching the trained palette cycle
// rather than by threading an offset down from the server layer. The case that
// matters most is the partial match: if the prompt carries a different number of
// image positions than the sidecar holds embeddings for, splicing anyway would
// misalign every row while still decoding into fluent text. That has to be a
// hard error, not a shrug.
#include "deepseek4/deepseek4_image_embed.h"

#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;

static void check(const Ds4ImageEmbed & e, const char * name,
                  std::vector<int32_t> toks, int64_t want_span, bool want_err) {
    std::string err;
    int64_t got = e.find_span(toks, &err);
    bool ok = (got == want_span) && (want_err == !err.empty());
    std::printf("  %-34s span=%3lld err=%d  %s\n", name, (long long) got,
                (int) !err.empty(), ok ? "ok" : "FAIL");
    if (!ok) fails++;
}

int main() {
    Ds4ImageEmbed e;
    e.active = true;
    e.n_embd = 4;
    e.n_tokens = 6;
    e.palette = {10, 20, 30};              // cycles to 10,20,30,10,20,30

    check(e, "no image",              {1, 2, 3, 4, 5},                   -1, false);
    check(e, "image at 0",            {10,20,30,10,20,30,7,8},            0, false);
    check(e, "image at 3",            {1,2,3,10,20,30,10,20,30},          3, false);
    check(e, "image at end",          {9,9,10,20,30,10,20,30},            2, false);
    check(e, "short run -> hard err", {1,10,20,30,10,99,9},              -1, true);
    check(e, "partial < one cycle",   {1,10,20,9,9},                     -1, false);
    check(e, "too short overall",     {10,20},                           -1, false);
    // A decoy that starts like the palette but breaks before completing a cycle
    // must not shadow the real image later in the sequence.
    check(e, "decoy then real image", {10,20,9,10,20,30,10,20,30},        3, false);

    Ds4ImageEmbed off;                     // inactive: never claims a span
    check(off, "inactive is inert",   {10,20,30,10,20,30},               -1, false);

    if (fails) {
        std::printf("\nFAILURES: %d\n", fails);
        return 1;
    }
    std::printf("\nall span cases pass\n");
    return 0;
}
