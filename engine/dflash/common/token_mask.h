// Abstract per-step logit mask for constrained decoding.
//
// The engine deliberately knows nothing about grammars, DSML, or xgrammar. It
// only advances the mask with each committed token and asks it to zero out
// disallowed logits — the same shape as the progress watchdog, which observes
// tokens without knowing what they mean. The implementation lives in
// src/backend/backend_dflash.cc, which already owns the tokenizer and is the
// only place that links xgrammar.
//
// Deciding WHEN to constrain is the mask's own business, not the engine's: it
// sees every token, so it tracks entry and exit of the tool-call region itself.
// Outside that region active() is false and generation is untouched, which is
// what keeps ordinary prose and reasoning entirely unconstrained.
#pragma once

#include <cstdint>

namespace dflash::common {

class TokenMask {
public:
    virtual ~TokenMask() = default;

    // A token was committed to the output stream. Speculative drafts that are
    // later rejected must NOT be reported here — only committed tokens.
    virtual void accept(int32_t token) = 0;

    // True when the mask currently constrains the next token.
    virtual bool active() const = 0;

    // Set the logits of disallowed tokens to -inf. Only called when active().
    // Must leave at least one token allowed; a mask that forbids everything
    // would make sampling undefined.
    virtual void apply(float *logits, int vocab) = 0;
};

}  // namespace dflash::common
