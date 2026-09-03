#include "pre_tokenizer.h"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

using dflash::common::PreTokenizer;
using dflash::common::pre_tokenize_text;
using dflash::common::pre_tokenizer_from_name;
using dflash::common::pre_tokenizer_name;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (condition) ++g_pass;                                            \
        else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", message); }    \
    } while (0)

static void expect_pieces(const char * input,
                          std::initializer_list<const char *> expected,
                          const char * message) {
    const std::vector<std::string> actual =
        pre_tokenize_text(input);
    std::vector<std::string> wanted;
    for (const char * piece : expected) wanted.emplace_back(piece);
    if (actual == wanted) {
        ++g_pass;
    } else {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n  actual:", message);
        for (const std::string & piece : actual)
            std::fprintf(stderr, " [%s]", piece.c_str());
        std::fprintf(stderr, "\n");
    }
    std::string joined;
    for (const std::string & piece : actual) joined += piece;
    CHECK(joined == input, "pre-tokenizer preserves every input byte");
}

int main() {
    PreTokenizer selected = PreTokenizer::JOYAI_LLM;
    CHECK(pre_tokenizer_from_name("joyai-llm", selected) &&
              selected == PreTokenizer::JOYAI_LLM,
          "joyai-llm metadata selects JoyAI");
    CHECK(!pre_tokenizer_from_name("joyai", selected),
          "unknown metadata cannot silently fall back");
    CHECK(std::string(pre_tokenizer_name(PreTokenizer::JOYAI_LLM)) ==
              "joyai-llm",
          "the diagnostic preserves its GGUF metadata spelling");

    // Regression from the production piper review. DS4 emits 003 and 379 as
    // whole BPE words; the old hard-coded splitter emitted single digits.
    expect_pieces("bip0032 BIP0038",
                  {"bip", "003", "2", " BIP", "003", "8"},
                  "JoyAI groups source-code digits in runs of at most three");
    expect_pieces("sed -n '379p'",
                  {"sed", " -", "n", " '", "379", "p", "'"},
                  "shell line numbers retain the DS4 JoyAI boundaries");
    expect_pieces("encryptBIP0038",
                  {"encryptBIP", "003", "8"},
                  "identifier digits use the production model's split");
    expect_pieces("1234567", {"123", "456", "7"},
                  "long digit strings split into consecutive triples");

    expect_pieces("#include <x>", {"#include", " <", "x", ">"},
                  "punctuation directly before letters stays attached");
    expect_pieces("\tname", {"\tname"},
                  "one non-newline prefix byte can lead a letter run");
    expect_pieces("    int x", {"   ", " int", " x"},
                  "indentation leaves one leading space for a word");
    expect_pieces(" >;\nnext", {" >;\n", "next"},
                  "punctuation keeps its trailing newline");
    expect_pieces("!?:\r\nnext", {"!?:\r\n", "next"},
                  "bare punctuation consumes a trailing CRLF");
    expect_pieces("  \n  code", {"  \n", " ", " code"},
                  "newline and post-newline indentation match DS4");
    expect_pieces(" \t ", {" \t "},
                  "terminal whitespace is preserved as one run");
    expect_pieces("中文かなカナ", {"中文かなカナ"},
                  "CJK, hiragana, and katakana form a contiguous run");
    expect_pieces(" café", {" café"},
                  "ordinary non-ASCII letters remain attached to the word");

    std::printf("pre-tokenizer: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
