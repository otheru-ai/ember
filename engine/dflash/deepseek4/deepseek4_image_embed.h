// Validation-only image-embedding path for the DeepSeek-V4-Flash vision graft.
//
// This is NOT the product multimodal path. It exists to answer one question --
// does DS4 produce image-grounded text from real projector output -- without
// first porting clip.cpp/libmtmd. The vision tower runs offline in Python
// (ds4-vision/tools/moonvit_forward.py, cross-checked against Moonshot's own
// implementation at 7e-6 relative error) and hands its 4096-d vectors here
// through a sidecar file.
//
// See docs/VISION-GRAFT.md. Enabled only when $EMBER_DS4_IMAGE_EMBED is set;
// one image, one request at a time, not concurrency-safe.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Ds4ImageEmbed {
    bool                 active   = false;
    int32_t              n_embd   = 0;
    int32_t              n_tokens = 0;
    std::vector<int32_t> palette;   // trained routing palette, in cycle order
    std::vector<float>   data;      // n_tokens * n_embd, n_embd fast axis

    // Process-wide instance, loaded once from $EMBER_DS4_IMAGE_EMBED.
    // `model_n_embd` is checked against the sidecar; a mismatch is fatal.
    static const Ds4ImageEmbed & instance(int64_t model_n_embd);

    // Locate the image span by matching the palette cycle.
    //
    // The palette is 64 fixed token IDs, so a run where tokens[s+k] ==
    // palette[k % 64] for k in [0, n_tokens) cannot plausibly arise from text.
    // Returns the start index, or -1 when no image is present.
    //
    // A run that starts but does not complete sets *err and returns -1: a
    // partial match means the prompt and the embeddings disagree about length,
    // and splicing anyway would misalign every subsequent image token while
    // still producing fluent output. That failure must be loud.
    int64_t find_span(const std::vector<int32_t> & tokens, std::string * err) const;

    // Overwrite the rows of one prefill chunk that fall inside the image span.
    // `chunk_start` is the absolute position of the chunk's first token.
    // No-op when inactive or when the chunk does not intersect the span.
    void splice(float * embed, int64_t chunk_start, int n_tok, int64_t span_start) const;
};

// C shim for src/server/main.c, which must emit the palette IDs at encode time.
// Keeping the sidecar parser on this side avoids a second implementation in C.
extern "C" {
// Returns the number of image tokens to emit (0 when no image is loaded) and,
// when non-zero, points *palette_out at n_palette_out fixed token IDs.
int ember_ds4_image_token_count(const int32_t ** palette_out, int * n_palette_out);
}
