#include "qwen4exp_shards.h"

#include <filesystem>
#include <regex>
#include <set>
#include <sys/stat.h>

namespace dflash::common {

bool qwen4exp_discover_gguf_shards(const char * model_path,
                                  std::vector<std::string> & paths,
                                  std::string & error) {
    paths.clear();
    error.clear();
    if (!model_path || model_path[0] == '\0') {
        error = "model path is empty";
        return false;
    }
    namespace fs = std::filesystem;
    const fs::path input(model_path);
    static const std::regex split_pattern(
        R"(^(.+)-([0-9]{5})-of-([0-9]{5})\.gguf$)",
        std::regex::ECMAScript);
    std::smatch match;
    const std::string filename = input.filename().string();
    if (!std::regex_match(filename, match, split_pattern)) {
        struct stat status{};
        if (::stat(model_path, &status) != 0 || status.st_size <= 0) {
            error = "failed to stat Qwen4Exp GGUF: " + std::string(model_path);
            return false;
        }
        paths.push_back(input.string());
        return true;
    }
    const unsigned long supplied = std::stoul(match[2].str());
    const unsigned long count = std::stoul(match[3].str());
    if (count == 0 || supplied == 0 || supplied > count) {
        error = "invalid canonical GGUF shard filename";
        return false;
    }
    const std::string stem = match[1].str();
    const fs::path directory = input.has_parent_path()
        ? input.parent_path() : fs::path(".");
    std::set<unsigned long> seen_ordinals;
    std::set<std::pair<dev_t, ino_t>> seen_files;
    std::error_code iterator_error;
    for (fs::directory_iterator it(directory, iterator_error), end;
         !iterator_error && it != end; it.increment(iterator_error)) {
        std::smatch sibling;
        const std::string sibling_name = it->path().filename().string();
        if (!std::regex_match(sibling_name, sibling, split_pattern) ||
            sibling[1].str() != stem) continue;
        const unsigned long sibling_count = std::stoul(sibling[3].str());
        if (sibling_count != count) {
            error = "mismatched filename shard count for GGUF stem " + stem;
            return false;
        }
        const unsigned long ordinal = std::stoul(sibling[2].str());
        if (ordinal == 0 || ordinal > count ||
            !seen_ordinals.insert(ordinal).second) {
            error = "duplicate or out-of-range GGUF shard filename";
            return false;
        }
    }
    if (iterator_error) {
        error = "failed to enumerate Qwen4Exp GGUF shard directory: " +
                iterator_error.message();
        return false;
    }
    if (seen_ordinals.size() != count) {
        error = "missing GGUF shard for canonical split set";
        return false;
    }
    paths.reserve(count);
    for (unsigned long ordinal = 1; ordinal <= count; ++ordinal) {
        std::string ordinal_text = std::to_string(ordinal);
        ordinal_text.insert(0, 5 - ordinal_text.size(), '0');
        // Reuse the regex-validated five-digit count text. This avoids a fixed
        // printf buffer and makes the formatter's bound explicit.
        const std::string shard_name = stem + "-" + ordinal_text + "-of-" +
                                       match[3].str() + ".gguf";
        const fs::path shard_path = directory / shard_name;
        struct stat status{};
        if (::stat(shard_path.c_str(), &status) != 0 || status.st_size <= 0) {
            error = "missing or empty Qwen4Exp GGUF shard: " +
                    shard_path.string();
            return false;
        }
        if (!seen_files.emplace(status.st_dev, status.st_ino).second) {
            error = "duplicate inode in Qwen4Exp GGUF shard set";
            return false;
        }
        paths.push_back(shard_path.string());
    }
    return true;
}

} // namespace dflash::common
