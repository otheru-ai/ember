// Default-off DeepSeek4 full-logit evidence probe.
//
// This is deliberately a standalone executable rather than a server option:
// no request can enable it and ordinary protocol output never acquires a
// vocabulary-sized diagnostic payload.

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#include "deepseek4/deepseek4_backend.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include "logits_probe_format.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifndef EMBER_CONFIGURED_GIT_HEAD
#error "ds4_logits_probe requires an embedded source revision"
#endif

namespace {

struct ProbeMode {
    const char *name;
    const char *role;
    int chunk;
    dflash::common::PrefillAttentionMode prefill_mode;
    bool force_exact;
};

bool parse_mode(const char *text, ProbeMode &mode) {
    if (std::strcmp(text, "exact-q1") == 0) {
        mode = {"exact-q1", "authority", 1,
                dflash::common::PrefillAttentionMode::Exact, true};
        return true;
    }
    if (std::strcmp(text, "exact-q4") == 0) {
        mode = {"exact-q4", "exact-batching-control", 4,
                dflash::common::PrefillAttentionMode::Exact, true};
        return true;
    }
    if (std::strcmp(text, "dense-q8") == 0) {
        mode = {"dense-q8", "shape-matched-approximate-diagnostic", 8,
                dflash::common::PrefillAttentionMode::Dense, false};
        return true;
    }
    return false;
}

bool unchanged_regular_file(const struct stat &before,
                            const struct stat &after) {
    return S_ISREG(before.st_mode) && S_ISREG(after.st_mode) &&
           before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
           before.st_size == after.st_size &&
           before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
           before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
           before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
           before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}

bool current_executable(std::string &path, std::string &error) {
    std::vector<char> buffer(4096U);
    for (;;) {
        const ssize_t count = readlink("/proc/self/exe", buffer.data(),
                                       buffer.size());
        if (count < 0) {
            error = "cannot resolve /proc/self/exe: " +
                    std::string(std::strerror(errno));
            return false;
        }
        if (static_cast<size_t>(count) < buffer.size()) {
            path.assign(buffer.data(), static_cast<size_t>(count));
            return true;
        }
        if (buffer.size() > 1024U * 1024U) {
            error = "executable path is unreasonably long";
            return false;
        }
        buffer.resize(buffer.size() * 2U);
    }
}

void usage(const char *program) {
    std::fprintf(stderr,
        "usage: %s {exact-q1|exact-q4|dense-q8} MODEL.gguf "
        "/absolute/output-dir TOKEN_ID [TOKEN_ID ...]\n",
        program);
}

bool reject_numeric_environment(std::string &error) {
    static constexpr const char *forbidden[] = {
        "DFLASH_EXPERT_BUDGET_MB",
        "DFLASH_DS4_FUSED_VERIFY_F16_KV",
        "LUCE_MMVQ_MAX_NCOLS",
        "DFLASH_DS4_SPEC_Q",
        "DFLASH_DS4_DRAFT",
        "DFLASH_DS4_SPEC",
    };
    for (const char *name : forbidden) {
        if (std::getenv(name) != nullptr) {
            error = "refusing numeric environment override: " +
                    std::string(name);
            return false;
        }
    }
    return true;
}

bool frozen_tokens(const std::vector<int32_t> &tokens) {
    static const std::vector<int32_t> expected = {
        1, 1000, 5000, 20000, 40000, 70000, 100000, 129279,
    };
    return tokens == expected;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        usage(argv[0]);
        return 2;
    }
    ProbeMode mode{};
    if (!parse_mode(argv[1], mode)) {
        usage(argv[0]);
        return 2;
    }
    const std::string model_path = argv[2];
    const std::string output_directory = argv[3];
    if (model_path.empty() || model_path[0] != '/') {
        std::fprintf(stderr, "model path must be absolute\n");
        return 2;
    }
    std::string error;
    if (!reject_numeric_environment(error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }

    std::vector<int32_t> token_ids;
    token_ids.reserve(static_cast<size_t>(argc - 4));
    for (int i = 4; i < argc; ++i) {
        int32_t token = -1;
        if (!ember_parse_logits_probe_token(argv[i], token, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 2;
        }
        token_ids.push_back(token);
    }
    if (token_ids.size() >
        static_cast<size_t>(std::numeric_limits<int>::max() - 1)) {
        std::fprintf(stderr, "token sequence is too long\n");
        return 2;
    }
    if (!frozen_tokens(token_ids)) {
        std::fprintf(stderr, "token ids differ from the frozen control sequence\n");
        return 2;
    }

    struct stat model_before {};
    if (stat(model_path.c_str(), &model_before) != 0 ||
        !S_ISREG(model_before.st_mode)) {
        std::fprintf(stderr, "model is not a readable regular file: %s\n",
                     model_path.c_str());
        return 1;
    }
    std::string model_digest;
    if (!ember_sha256_regular_file(model_path, model_digest, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    std::string executable_path;
    std::string executable_digest;
    if (!current_executable(executable_path, error) ||
        !ember_sha256_regular_file(executable_path, executable_digest, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    const char *exact_chunk = mode.chunk == 4 ? "4" : "1";
    if (setenv("DFLASH_DS4_EXACT_PREFILL_CHUNK", exact_chunk, 1) != 0 ||
        setenv("DFLASH_DS4_EXACT_PREFILL_SKIP_LOGITS", "1", 1) != 0) {
        std::fprintf(stderr, "cannot pin diagnostic prefill controls: %s\n",
                     std::strerror(errno));
        return 1;
    }

    dflash::common::DeepSeek4BackendConfig config;
    config.model_path = model_path.c_str();
    config.chunk = mode.chunk;
    config.prefill_mode = mode.prefill_mode;
    config.max_ctx = std::max(32, static_cast<int>(token_ids.size()) + 1);
    // Force a monolithic load. The diagnostic is a control for Ember's target
    // computation, not the optional CPU expert streaming path.
    config.fused_decode = true;
    dflash::common::DeepSeek4Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "cannot initialize DeepSeek4 control model\n");
        return 1;
    }
    std::vector<float> logits;
    int effective_prefill_chunk = 0;
    if (!backend.diagnostic_next_logits(
            token_ids, mode.force_exact, logits, effective_prefill_chunk,
            error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (effective_prefill_chunk != mode.chunk) {
        std::fprintf(stderr,
            "effective prefill chunk differs from requested mode: %d != %d\n",
            effective_prefill_chunk, mode.chunk);
        return 1;
    }
    if (logits.size() != 129280U) {
        std::fprintf(stderr,
            "model vocabulary differs from frozen control: %zu != 129280\n",
            logits.size());
        return 1;
    }

    struct stat model_after {};
    if (stat(model_path.c_str(), &model_after) != 0 ||
        !unchanged_regular_file(model_before, model_after)) {
        std::fprintf(stderr, "model identity changed during the probe\n");
        return 1;
    }

    EmberLogitsProbeBundle bundle;
    bundle.model_path = model_path;
    bundle.model_sha256 = model_digest;
    bundle.binary_sha256 = executable_digest;
    bundle.ember_revision = EMBER_CONFIGURED_GIT_HEAD;
    bundle.prefill_mode = mode.name;
    bundle.comparison_role = mode.role;
    bundle.prefill_chunk = effective_prefill_chunk;
    bundle.token_ids = token_ids;
    bundle.logits = std::move(logits);
    if (!ember_write_logits_probe_bundle(output_directory, bundle, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr,
        "[ds4-logits-probe] wrote bound next-token row: dir=%s vocab=%zu\n",
        output_directory.c_str(), bundle.logits.size());
    return 0;
}
