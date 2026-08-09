// End-to-end: does the grammar ember generates actually constrain a real
// constrained-decoding engine?
//
// test_tool_grammar.c asserts on the emitted EBNF text, which proves the text
// is what we meant to write, not that a parser agrees. This compiles the same
// generator output with the vendored xgrammar core and checks the shapes
// production emits are genuinely unrepresentable.
//
// Hermetic by design: a synthetic byte vocabulary rather than the model's, so
// this runs in the GPU-free gauntlet with no GGUF present. Vocabulary choice
// does not affect GrammarMatcher::AcceptString, which walks the grammar
// directly; it only affects token bitmasks.
#include <cstdio>
#include <string>
#include <vector>

#include <xgrammar/xgrammar.h>

#include "../src/model/tool_grammar.h"

static int passed = 0;
static int failed = 0;

#define CHECK(expr) do { \
    if (expr) { ++passed; } \
    else { ++failed; std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                                  __FILE__, __LINE__, #expr); } \
} while (0)

// U+FF5C FULLWIDTH VERTICAL LINE, the DSML delimiter (matches chat_template.c).
static std::string P(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '|') out += "\xef\xbd\x9c";
        else out += c;
    }
    return out;
}

static bool accepts(const xgrammar::CompiledGrammar &cg, const std::string &s) {
    xgrammar::GrammarMatcher m(cg);
    for (char c : s) {
        if (!m.AcceptString(std::string(1, c))) return false;
    }
    return true;
}

int main() {
    // terminal requires `command`; browser_back genuinely declares no
    // properties and is the one tool for which an empty invoke is legal.
    const char *tools =
        "[{\"type\":\"function\",\"function\":{\"name\":\"terminal\","
        "\"parameters\":{\"type\":\"object\",\"required\":[\"command\"],"
        "\"properties\":{\"command\":{\"type\":\"string\"}}}}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"browser_back\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";

    char *ebnf = ember_tool_grammar_build(tools, true);
    CHECK(ebnf != nullptr);
    if (!ebnf) return 1;

    xgrammar::Grammar g = xgrammar::Grammar::FromEBNF(ebnf);

    // A byte vocabulary is enough to compile against; see the header comment.
    std::vector<std::string> vocab;
    vocab.reserve(256);
    for (int i = 0; i < 256; ++i) {
        vocab.push_back(std::string(1, static_cast<char>(i)));
    }
    xgrammar::TokenizerInfo ti(vocab, xgrammar::VocabType::RAW,
                               static_cast<int>(vocab.size()));
    xgrammar::GrammarCompiler compiler(ti);
    xgrammar::CompiledGrammar cg = compiler.CompileGrammar(g);

    const std::string blk  = P("<|DSML|tool_calls>");
    const std::string term = blk + P("<|DSML|invoke name=\"terminal\">");

    // The production failure: a complete <invoke> with no <parameter> at all
    // ("$ is missing required property command"). Must be unrepresentable.
    CHECK(!accepts(cg, term + P("</|DSML|invoke>")));

    // The other observed rejection variant: nested invoke.
    CHECK(!accepts(cg, term + P("<|DSML|invoke name=\"terminal\">")));

    // A correct call must still be reachable, or the grammar is useless.
    CHECK(accepts(cg,
        term + P("<|DSML|parameter name=\"command\" string=\"true\">ls")));

    // ...and a legitimately parameterless tool must not be over-constrained.
    CHECK(accepts(cg, blk + P("<|DSML|invoke name=\"browser_back\">") +
                      P("</|DSML|invoke>")));

    // A string-typed property sent in the JSON form is refused.
    CHECK(!accepts(cg,
        term + P("<|DSML|parameter name=\"command\" string=\"false\">")));

    // ── the mask mechanics backend_dflash.cc depends on ──────────────────
    // Same DLTensor setup and bit convention as DsmlGrammarMask::apply(). If
    // the bit sense were inverted this would pass tokens the grammar forbids,
    // which is exactly the failure that would be invisible at runtime.
    {
        xgrammar::GrammarMatcher m(cg, std::nullopt,
                                   /*terminate_without_stop_token=*/true);
        CHECK(m.AcceptString(term));  // opened terminal, nothing emitted yet

        std::vector<int32_t> bits((size_t)xgrammar::GetBitmaskSize(256));
        int64_t shape[1] = {(int64_t)bits.size()};
        DLTensor t{};
        t.data = bits.data();
        t.device = DLDevice{kDLCPU, 0};
        t.ndim = 1;
        t.dtype = xgrammar::GetBitmaskDLType();
        t.shape = shape;
        t.strides = nullptr;
        t.byte_offset = 0;
        m.FillNextTokenBitmask(&t);

        auto allowed = [&](unsigned char c) {
            return (bits[(size_t)(c >> 5)] >> (c & 31)) & 1;
        };

        // The only legal continuations are a <parameter> tag or the `ws` the
        // renderer emits between tags, so '<' and whitespace are allowed while
        // ordinary text is not.
        CHECK(allowed('<'));
        CHECK(allowed(' '));   // ws ::= [ \t\n\r]* -- chat_template.c newlines
        CHECK(!allowed('Z'));
        CHECK(!allowed('{'));

        // The mask must never forbid everything, or sampling is undefined.
        bool any = false;
        for (int32_t w : bits) if (w) { any = true; break; }
        CHECK(any);
    }

    // ── merged-delimiter token across the literal boundary ───────────────
    // Region entry matches the marker WITHOUT its trailing '>' (the model
    // emits ">\n" as one token, so the bare '>' never appears -- see
    // backend_dflash.cc set_tool_region_ids). The mask therefore replays only
    // the prefix into a fresh matcher, leaving it mid-literal. The very next
    // real token spans the literal's final '>' AND the ws that follows, so the
    // grammar must accept a token crossing that boundary or constrained
    // decoding dies on the first token of every tool call.
    {
        std::vector<std::string> v2;
        v2.reserve(257);
        for (int i = 0; i < 256; ++i) v2.push_back(std::string(1, (char)i));
        v2.push_back(">\n");                       // the merged token
        const int merged_id = 256;

        xgrammar::TokenizerInfo ti2(v2, xgrammar::VocabType::RAW,
                                    static_cast<int>(v2.size()));
        xgrammar::CompiledGrammar cg2 =
            xgrammar::GrammarCompiler(ti2).CompileGrammar(g);

        xgrammar::GrammarMatcher m(cg2, std::nullopt, true);
        // Everything up to but excluding the '>' -- what the prefix match sees.
        const std::string prefix = P("<|DSML|tool_calls");
        CHECK(m.AcceptString(prefix));
        // ">\n" must be accepted as a single token here.
        CHECK(m.AcceptToken(merged_id));
        // ...and the block must still be completable afterwards.
        CHECK(m.AcceptString(P("<|DSML|invoke name=\"browser_back\">")));
    }

    // ── discriminated (if/then) requirements ─────────────────────────────
    // patch requires only `mode`, but requires path+old_string+new_string once
    // mode="replace". Production emitted {mode, new_string} 17 times -- valid
    // against the flat schema, refused by the tool every time. The grammar
    // turns each mode into its own branch, so that shape cannot be derived.
    {
        const char *ptools =
            "[{\"type\":\"function\",\"function\":{\"name\":\"patch\","
            "\"parameters\":{\"type\":\"object\",\"required\":[\"mode\"],"
            "\"properties\":{"
            "\"mode\":{\"type\":\"string\",\"enum\":[\"replace\",\"patch\"]},"
            "\"path\":{\"type\":\"string\"},"
            "\"old_string\":{\"type\":\"string\"},"
            "\"new_string\":{\"type\":\"string\"},"
            "\"patch\":{\"type\":\"string\"}},"
            "\"allOf\":["
            "{\"if\":{\"properties\":{\"mode\":{\"const\":\"replace\"}}},"
            " \"then\":{\"required\":[\"path\",\"old_string\",\"new_string\"]}},"
            "{\"if\":{\"properties\":{\"mode\":{\"const\":\"patch\"}}},"
            " \"then\":{\"required\":[\"patch\"]}}]}}}]";

        char *pe = ember_tool_grammar_build(ptools, true);
        CHECK(pe != nullptr);
        xgrammar::Grammar pg = xgrammar::Grammar::FromEBNF(pe);
        xgrammar::CompiledGrammar pcg =
            xgrammar::GrammarCompiler(ti).CompileGrammar(pg);

        const std::string open = P("<|DSML|tool_calls>") +
                                 P("<|DSML|invoke name=\"patch\">");
        auto param = [&](const char *n, const char *v) {
            return P(std::string("<|DSML|parameter name=\"") + n +
                     "\" string=\"true\">") + v + P("</|DSML|parameter>");
        };

        // THE PRODUCTION BUG: mode=replace then straight to a non-required
        // property, skipping path.
        CHECK(!accepts(pcg, open + param("mode", "replace") +
                            param("new_string", "x")));
        // Closing the invoke right after the discriminator is likewise dead.
        CHECK(!accepts(pcg, open + param("mode", "replace") +
                            P("</|DSML|invoke>")));
        // The correct replace call is still reachable.
        CHECK(accepts(pcg, open + param("mode", "replace") +
                           param("path", "/f.c") + param("old_string", "a") +
                           param("new_string", "b")));
        // ...and the other branch requires its own property instead.
        CHECK(accepts(pcg, open + param("mode", "patch") +
                           param("patch", "*** Begin Patch")));
        CHECK(!accepts(pcg, open + param("mode", "patch") +
                            P("</|DSML|invoke>")));
        // A branch cannot borrow the other's obligations.
        CHECK(!accepts(pcg, open + param("mode", "patch") +
                            param("path", "/f.c")));
        free(pe);
    }

    // ── '<' inside a value ───────────────────────────────────────────────
    // strval was [^<]*, which made every value containing '<' unemittable.
    // Production hit it at once: the model tried to patch C code, was forced
    // to close the parameter at the '<', got truncated content, and looped.
    // It diagnosed the cause itself: "The patch keeps truncating at `<`".
    {
        const std::string open = blk + P("<|DSML|invoke name=\"terminal\">");
        auto cmd = [&](const std::string &v) {
            return open + P("<|DSML|parameter name=\"command\" string=\"true\">") + v;
        };
        // The exact text from the stuck turn.
        CHECK(accepts(cg, cmd("for (int i = 0; i < head_len && path[i]; ++i)")));
        CHECK(accepts(cg, cmd("#include <stdio.h>")));
        CHECK(accepts(cg, cmd("if (a<b && c<=d) return;")));
        CHECK(accepts(cg, cmd("echo done > /tmp/f && test $x -lt 3")));
        // A value may legitimately END with '<'.
        CHECK(accepts(cg, cmd("truncated at <")));
        // ...and the closing tag must still be found, not swallowed by the value.
        CHECK(accepts(cg, cmd("i < n") + P("</|DSML|parameter>") +
                          P("</|DSML|invoke>") + P("</|DSML|tool_calls>")));
    }

    // A discriminator value carrying no conditional must remain callable.
    // Emitting only branch values would forbid skill_manage("delete") etc.
    {
        const char *st =
            "[{\"function\":{\"name\":\"skill_manage\",\"parameters\":{"
            "\"type\":\"object\",\"required\":[\"action\",\"name\"],"
            "\"properties\":{"
            "\"action\":{\"type\":\"string\","
            "  \"enum\":[\"patch\",\"delete\"]},"
            "\"name\":{\"type\":\"string\"},"
            "\"old_string\":{\"type\":\"string\"},"
            "\"new_string\":{\"type\":\"string\"}},"
            "\"allOf\":[{\"if\":{\"properties\":{\"action\":{\"const\":\"patch\"}}},"
            " \"then\":{\"required\":[\"old_string\",\"new_string\"]}}]}}}]";
        char *e = ember_tool_grammar_build(st, true);
        CHECK(e != nullptr);
        xgrammar::CompiledGrammar cg2 =
            xgrammar::GrammarCompiler(ti).CompileGrammar(
                xgrammar::Grammar::FromEBNF(e));
        const std::string open = blk + P("<|DSML|invoke name=\"skill_manage\">");
        auto pr = [&](const char *n, const char *v) {
            return P(std::string("<|DSML|parameter name=\"") + n +
                     "\" string=\"true\">") + v + P("</|DSML|parameter>");
        };
        // delete has no conditional -- action + name must suffice.
        CHECK(accepts(cg2, open + pr("action","delete") + pr("name","x") +
                           P("</|DSML|invoke>")));
        // patch still requires old_string/new_string.
        CHECK(!accepts(cg2, open + pr("action","patch") + pr("name","x") +
                            P("</|DSML|invoke>")));
        CHECK(accepts(cg2, open + pr("action","patch") + pr("name","x") +
                           pr("old_string","a") + pr("new_string","b")));
        free(e);
    }

    free(ebnf);
    std::printf("tool grammar match: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
