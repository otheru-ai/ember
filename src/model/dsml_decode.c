// B6 — DSML decode-state tracker. See dsml_decode.h. Ported from ds4's
// dsml_decode_tracker (ds4_server.c); ember's ember_dsml_syntax field names
// (calls_open/close, invoke_open/close, param_open/close) replace ds4's.
#include "dsml_decode.h"

#include <ctype.h>
#include <string.h>

// ── literal-match primitives (ds4 raw_full_lit / raw_partial_lit / find_lit) ──
static bool full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    return pos <= raw_len && raw_len - pos >= n && !memcmp(raw + pos, lit, n);
}

// A truncated prefix of `lit` sits at raw[pos:raw_len] (the tag may still be
// mid-emission); treat that as structure so we don't sample into a partial tag.
static bool partial_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos > raw_len || raw_len - pos >= n) return false;
    return !memcmp(raw + pos, lit, raw_len - pos);
}

static bool partial_lit_min(const char *raw, size_t raw_len, size_t pos,
                            const char *lit, size_t min_len) {
    size_t n = strlen(lit);
    if (!raw || pos > raw_len || raw_len - pos >= n) return false;
    size_t avail = raw_len - pos;
    return avail >= min_len && !memcmp(raw + pos, lit, avail);
}

static const char *find_lit_bounded(const char *s, size_t n, const char *lit) {
    size_t m = strlen(lit);
    if (m == 0) return s;
    if (n < m) return NULL;
    for (size_t i = 0; i <= n - m; i++)
        if (!memcmp(s + i, lit, m)) return s + i;
    return NULL;
}

static size_t max_tool_start_len(void) {
    int n = 0;
    const ember_dsml_syntax *sx = ember_dsml_syntaxes(&n);
    size_t max = 0;
    for (int i = 0; i < n; i++) {
        size_t k = strlen(sx[i].calls_open);
        if (k > max) max = k;
    }
    return max;
}

// Earliest tool_calls opener across all spellings within raw[start:raw_len].
static bool find_tool_start_from(const char *raw, size_t raw_len, size_t start,
                                 size_t *pos_out, const ember_dsml_syntax **syn_out) {
    if (start > raw_len) return false;
    int n = 0;
    const ember_dsml_syntax *sx = ember_dsml_syntaxes(&n);
    const char *best = NULL;
    const ember_dsml_syntax *best_syn = NULL;
    for (int i = 0; i < n; i++) {
        const char *p = find_lit_bounded(raw + start, raw_len - start, sx[i].calls_open);
        if (p && (!best || p < best)) { best = p; best_syn = &sx[i]; }
    }
    if (!best) return false;
    *pos_out = (size_t)(best - raw) + strlen(best_syn->calls_open);
    *syn_out = best_syn;
    return true;
}

// Does the parameter tag at raw[tag_start:tag_end] carry string="true"?
static bool attr_is_string_true(const char *raw, size_t raw_len,
                                size_t tag_start, size_t tag_end) {
    if (tag_end <= tag_start || tag_end > raw_len) return false;
    const char *tag = raw + tag_start;
    size_t taglen = tag_end - tag_start;
    // find `string="` within the tag, then compare the quoted value to `true`.
    static const char pat[] = "string=\"";
    for (size_t i = 0; i + sizeof(pat) - 1 <= taglen; i++) {
        if (!memcmp(tag + i, pat, sizeof(pat) - 1)) {
            const char *v = tag + i + sizeof(pat) - 1;
            size_t rem = taglen - (i + sizeof(pat) - 1);
            return rem >= 5 && !memcmp(v, "true\"", 5);
        }
    }
    return false;
}

bool ember_dsml_force_greedy(const ember_dsml_tracker *dt) {
    if (!dt || dt->decode == EMBER_DSML_OUTSIDE) return false;       // not in a call
    // payload positions (string body / JSON string value) sample at temperature.
    return dt->decode != EMBER_DSML_STRING_BODY &&
           dt->decode != EMBER_DSML_JSON_STRING;
}

void ember_dsml_tracker_init(ember_dsml_tracker *dt) {
    memset(dt, 0, sizeof(*dt));
    dt->mode = EMBER_DSML_TRK_SEARCH;
    dt->decode = EMBER_DSML_OUTSIDE;
}

void ember_dsml_tracker_update(ember_dsml_tracker *dt, const char *raw, size_t raw_len) {
    if (!dt || !raw) return;

    for (;;) {
        if (dt->mode == EMBER_DSML_TRK_DONE) {
            dt->decode = EMBER_DSML_OUTSIDE;
            return;
        }

        if (dt->mode == EMBER_DSML_TRK_SEARCH) {
            size_t pos = 0;
            const ember_dsml_syntax *syn = NULL;
            if (!find_tool_start_from(raw, raw_len, dt->pos, &pos, &syn)) {
                size_t hold = max_tool_start_len();
                dt->pos = raw_len > hold ? raw_len - hold : 0;
                dt->decode = EMBER_DSML_OUTSIDE;
                return;
            }
            dt->syn = syn;
            dt->pos = pos;
            dt->mode = EMBER_DSML_TRK_STRUCTURAL;
            dt->decode = EMBER_DSML_STRUCTURAL;
        }

        if (dt->mode == EMBER_DSML_TRK_STRING_BODY) {
            while (dt->pos < raw_len) {
                if (full_lit(raw, raw_len, dt->pos, dt->syn->param_close)) {
                    dt->pos += strlen(dt->syn->param_close);
                    dt->mode = EMBER_DSML_TRK_STRUCTURAL;
                    dt->decode = EMBER_DSML_STRUCTURAL;
                    goto structural;
                }
                if (partial_lit_min(raw, raw_len, dt->pos, dt->syn->param_close, 2)) {
                    dt->decode = EMBER_DSML_STRUCTURAL;
                    return;
                }
                dt->pos++;
            }
            dt->decode = EMBER_DSML_STRING_BODY;
            return;
        }

        if (dt->mode == EMBER_DSML_TRK_JSON_PARAM) {
            while (dt->pos < raw_len) {
                if (!dt->json_in_string) {
                    if (full_lit(raw, raw_len, dt->pos, dt->syn->param_close)) {
                        dt->pos += strlen(dt->syn->param_close);
                        dt->mode = EMBER_DSML_TRK_STRUCTURAL;
                        dt->decode = EMBER_DSML_STRUCTURAL;
                        goto structural;
                    }
                    if (partial_lit_min(raw, raw_len, dt->pos, dt->syn->param_close, 2)) {
                        dt->decode = EMBER_DSML_STRUCTURAL;
                        return;
                    }
                }
                unsigned char c = (unsigned char)raw[dt->pos++];
                if (dt->json_in_string) {
                    if (dt->json_escaped)      dt->json_escaped = false;
                    else if (c == '\\')        dt->json_escaped = true;
                    else if (c == '"')         dt->json_in_string = false;
                } else if (c == '"') {
                    dt->json_in_string = true;
                }
            }
            dt->decode = dt->json_in_string ? EMBER_DSML_JSON_STRING
                                            : EMBER_DSML_JSON_STRUCTURAL;
            return;
        }

structural:
        while (dt->mode == EMBER_DSML_TRK_STRUCTURAL) {
            while (dt->pos < raw_len && isspace((unsigned char)raw[dt->pos])) dt->pos++;
            if (dt->pos >= raw_len) {
                dt->decode = EMBER_DSML_STRUCTURAL;
                return;
            }

            if (full_lit(raw, raw_len, dt->pos, dt->syn->calls_close)) {
                dt->mode = EMBER_DSML_TRK_DONE;
                dt->pos += strlen(dt->syn->calls_close);
                dt->decode = EMBER_DSML_OUTSIDE;
                return;
            }
            if (full_lit(raw, raw_len, dt->pos, dt->syn->invoke_close)) {
                dt->pos += strlen(dt->syn->invoke_close);
                continue;
            }
            if (full_lit(raw, raw_len, dt->pos, dt->syn->invoke_open)) {
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) { dt->decode = EMBER_DSML_STRUCTURAL; return; }
                dt->pos = (size_t)(tag_end - raw) + 1;
                continue;
            }
            if (full_lit(raw, raw_len, dt->pos, dt->syn->param_open)) {
                size_t tag_start = dt->pos;
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) { dt->decode = EMBER_DSML_STRUCTURAL; return; }
                size_t tag_after = (size_t)(tag_end - raw) + 1;
                bool string_value = attr_is_string_true(raw, raw_len, tag_start, tag_after);
                dt->pos = tag_after;
                if (string_value) {
                    dt->mode = EMBER_DSML_TRK_STRING_BODY;
                    dt->decode = EMBER_DSML_STRING_BODY;
                } else {
                    dt->mode = EMBER_DSML_TRK_JSON_PARAM;
                    dt->json_in_string = false;
                    dt->json_escaped = false;
                    dt->decode = EMBER_DSML_JSON_STRUCTURAL;
                }
                break;
            }

            // A tag opener/closer may be mid-emission: hold structural.
            if (partial_lit(raw, raw_len, dt->pos, dt->syn->calls_close) ||
                partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_open) ||
                partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_close) ||
                partial_lit(raw, raw_len, dt->pos, dt->syn->param_open) ||
                partial_lit(raw, raw_len, dt->pos, dt->syn->param_close)) {
                dt->decode = EMBER_DSML_STRUCTURAL;
                return;
            }

            dt->decode = EMBER_DSML_STRUCTURAL;
            return;
        }
    }
}
