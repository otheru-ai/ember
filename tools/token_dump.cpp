// Metadata-only tokenizer probe for production GGUFs.
//
// This deliberately links only the tokenizer and ggml-base: it never creates a
// model backend, allocates tensors, or touches the GPU.  Tokenizer parity bugs
// otherwise hide inside a 100 GB server startup and are difficult to compare
// byte-for-byte with ds4's CPU-only tokenizer utility.

#include "tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using dflash::common::Tokenizer;

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf [--text TEXT | --file PATH]\n",
                 argv0);
}

static bool read_file(const char * path, std::string & out) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot open input file: %s\n", path);
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(input),
               std::istreambuf_iterator<char>());
    if (input.bad()) {
        std::fprintf(stderr, "cannot read input file: %s\n", path);
        return false;
    }
    return true;
}

static void print_hex(const std::string & bytes) {
    static const char digits[] = "0123456789abcdef";
    for (unsigned char byte : bytes) {
        std::putchar(digits[byte >> 4]);
        std::putchar(digits[byte & 0x0f]);
    }
}

int main(int argc, char ** argv) {
    if (argc != 4 || (std::string(argv[2]) != "--text" &&
                      std::string(argv[2]) != "--file")) {
        usage(argv[0]);
        return 2;
    }

    std::string input;
    if (std::string(argv[2]) == "--file") {
        if (!read_file(argv[3], input)) return 1;
    } else {
        input = argv[3];
    }

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(argv[1])) return 1;

    const std::vector<int32_t> ids = tokenizer.encode(input);
    std::printf("count=%zu\nids=", ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) std::putchar(',');
        std::printf("%d", ids[i]);
    }
    std::putchar('\n');

    for (size_t i = 0; i < ids.size(); ++i) {
        const std::string bytes = tokenizer.token_text(ids[i]);
        std::printf("token[%zu] id=%d bytes=", i, ids[i]);
        print_hex(bytes);
        std::putchar('\n');
    }
    return 0;
}
