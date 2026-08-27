#include "gguf_quantize_common.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <iomanip>
#include <utility>

namespace ember::gguf_quantize {
namespace {

constexpr const char * kFormat = "Q4_0_ROCMI4";

bool starts_with(const std::string & value, const char * prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string & value, const char * suffix) {
    const std::string ending(suffix);
    return value.size() >= ending.size() &&
           value.compare(value.size() - ending.size(), ending.size(), ending) == 0;
}

bool contains(const std::string & value, const char * needle) {
    return value.find(needle) != std::string::npos;
}

bool parse_tensor_override(const std::string & argument,
                           TensorTypeOverride & override,
                           std::string & error) {
    const std::size_t separator = argument.rfind('=');
    if (separator == std::string::npos || separator == 0) {
        error = "--tensor-type requires REGEX={Q4_0_ROCMI4,Q6_K,Q4_0_ROCMFP4_FAST}";
        return false;
    }
    const std::string format = argument.substr(separator + 1);
    if (format == "Q4_0_ROCMI4") {
        override.format = TensorFormat::rocmi4;
    } else if (format == "Q6_K") {
        override.format = TensorFormat::q6_k;
    } else if (format == "Q4_0_ROCMFP4_FAST") {
        override.format = TensorFormat::rocmfp4_fast;
    } else {
        error = "--tensor-type requires REGEX={Q4_0_ROCMI4,Q6_K,Q4_0_ROCMFP4_FAST}";
        return false;
    }
    override.pattern = argument.substr(0, separator);
    return true;
}

bool parse_u64(const std::string & argument, const char * option,
               std::uint64_t & result, std::string & error) {
    if (argument.empty() || argument.front() == '-') {
        error = std::string(option) + " requires a non-negative integer";
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(argument.c_str(), &end, 10);
    if (errno != 0 || end == argument.c_str() || *end != '\0') {
        error = std::string(option) + " requires a non-negative integer";
        return false;
    }
    result = static_cast<std::uint64_t>(parsed);
    return true;
}

}  // namespace

bool parse_options(int argc, const char * const * argv,
                   Options & options, std::string & error) {
    options = Options{};
    error.clear();
    if (argc == 2 && std::string(argv[1]) == "--build-info-json") {
        options.build_info_json = true;
        return true;
    }
    if (argc < 2) {
        error = "missing arguments";
        return false;
    }

    std::vector<std::string> positional;
    bool saw_positional = false;
    bool saw_budget = false;
    bool saw_reserve = false;
    bool saw_intervention = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (starts_with(argument, "--")) {
            if (saw_positional) {
                error = "all options must precede positional arguments";
                return false;
            }
            if (argument == "--keep-split") {
                options.keep_split = true;
            } else if (argument == "--dry-size-json") {
                options.dry_size_json = true;
            } else if (argument == "--device-budget-bytes" ||
                       argument == "--runtime-reserve-bytes") {
                const bool is_budget = argument == "--device-budget-bytes";
                if (++index >= argc) {
                    error = argument + " requires an argument";
                    return false;
                }
                std::uint64_t value = 0;
                if (!parse_u64(argv[index], argument.c_str(), value, error)) {
                    return false;
                }
                if (is_budget) {
                    if (saw_budget) {
                        error = "--device-budget-bytes may be specified only once";
                        return false;
                    }
                    saw_budget = true;
                    options.device_budget_bytes = value;
                } else {
                    if (saw_reserve) {
                        error = "--runtime-reserve-bytes may be specified only once";
                        return false;
                    }
                    saw_reserve = true;
                    options.runtime_reserve_bytes = value;
                }
            } else if (argument == "--tensor-type") {
                if (++index >= argc) {
                    error = "--tensor-type requires an argument";
                    return false;
                }
                TensorTypeOverride override;
                if (!parse_tensor_override(argv[index], override, error)) {
                    return false;
                }
                options.tensor_type_overrides.push_back(std::move(override));
            } else if (argument == "--intervention-manifest") {
                if (saw_intervention) {
                    error = "--intervention-manifest may be specified only once";
                    return false;
                }
                if (++index >= argc || argv[index][0] == '\0') {
                    error = "--intervention-manifest requires an argument";
                    return false;
                }
                saw_intervention = true;
                options.intervention_manifest_path = argv[index];
            } else {
                error = "unknown option: " + argument;
                return false;
            }
        } else {
            saw_positional = true;
            positional.push_back(argument);
        }
    }

    if (positional.size() != 4) {
        error = "expected INPUT OUTPUT Q4_0_ROCMI4 THREADS";
        return false;
    }
    if (positional[2] != kFormat) {
        error = "only Q4_0_ROCMI4 output is supported";
        return false;
    }
    if (options.tensor_type_overrides.empty()) {
        error = "at least one --tensor-type REGEX=Q4_0_ROCMI4 override is required";
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(positional[3].c_str(), &end, 10);
    if (errno != 0 || end == positional[3].c_str() || *end != '\0' ||
        parsed == 0 || parsed > 1024 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        error = "THREADS must be an integer in [1, 1024]";
        return false;
    }
    options.input_path = positional[0];
    options.output_path = positional[1];
    options.threads = static_cast<std::size_t>(parsed);
    if (saw_budget != saw_reserve) {
        error = "--device-budget-bytes and --runtime-reserve-bytes must be supplied together";
        return false;
    }
    options.budget_configured = saw_budget;
    if (options.dry_size_json && !options.budget_configured) {
        error = "--dry-size-json requires explicit device budget and runtime reserve bytes";
        return false;
    }
    return true;
}

const char * tensor_format_name(TensorFormat format) {
    switch (format) {
        case TensorFormat::rocmi4: return "Q4_0_ROCMI4";
        case TensorFormat::q6_k: return "Q6_K";
        case TensorFormat::rocmfp4_fast: return "Q4_0_ROCMFP4_FAST";
    }
    return "unknown";
}

bool tensor_allows_quantization(const std::string & name, std::size_t rank) {
    if (rank < 2 || !ends_with(name, "weight")) {
        return false;
    }
    if (contains(name, "norm") || name == "output.weight" ||
        contains(name, "router") || contains(name, "ffn_gate_inp.weight") ||
        contains(name, "conv") || contains(name, "position") ||
        contains(name, "pos_embd") || contains(name, "rel_pos") ||
        contains(name, "patch")) {
        return false;
    }
    return true;
}

std::string build_info_json(const std::string & ember_revision) {
    return "{\"tool\":\"ember-gguf-quantize\","
           "\"ember_revision\":\"" + ember_revision + "\","
           "\"rocmfpx_revision\":\"c49ebdbd5c9f01ec242369f9e7f7967855f80cba\","
           "\"format\":\"Q4_0_ROCMI4\",\"ggml_tensor_type\":108,"
           "\"per_tensor_formats\":[\"Q4_0_ROCMI4\",\"Q6_K\",\"Q4_0_ROCMFP4_FAST\"],"
           "\"intervention_manifest_schema\":1}";
}

std::string size_report_json(const SizeReport & report) {
    constexpr double gib = 1073741824.0;
    const std::uint64_t total = report.artifact_bytes >
            std::numeric_limits<std::uint64_t>::max() - report.runtime_reserve_bytes
        ? std::numeric_limits<std::uint64_t>::max()
        : report.artifact_bytes + report.runtime_reserve_bytes;
    const bool nonnegative = report.budget_bytes >= total;
    const std::uint64_t difference = nonnegative
        ? report.budget_bytes - total : total - report.budget_bytes;
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\"artifact_bytes\":" << report.artifact_bytes
           << ",\"artifact_gib\":" << static_cast<double>(report.artifact_bytes) / gib
           << ",\"shard_count\":" << report.shard_bytes.size()
           << ",\"shard_bytes\":[";
    for (std::size_t index = 0; index < report.shard_bytes.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << report.shard_bytes[index];
    }
    output << ']'
           << ",\"runtime_reserve_bytes\":" << report.runtime_reserve_bytes
           << ",\"runtime_reserve_gib\":" << static_cast<double>(report.runtime_reserve_bytes) / gib
           << ",\"budget_bytes\":" << report.budget_bytes
           << ",\"budget_gib\":" << static_cast<double>(report.budget_bytes) / gib
           << ",\"total_bytes\":" << total
           << ",\"total_gib\":" << static_cast<double>(total) / gib
           << ",\"headroom_bytes\":" << (nonnegative ? "" : "-") << difference
           << ",\"headroom_gib\":" << (nonnegative ? "" : "-")
           << static_cast<double>(difference) / gib
           << ",\"fits\":" << (report.fits ? "true" : "false");
    if (report.intervention_validated) {
        output << ",\"intervention_manifest_sha256\":\""
               << report.intervention_manifest_sha256 << "\""
               << ",\"intervention_target_names_sha256\":\""
               << report.intervention_target_names_sha256 << "\""
               << ",\"intervention_target_count\":"
               << report.intervention_targets.size()
               << ",\"intervention_targets\":[";
        for (std::size_t index = 0; index < report.intervention_targets.size(); ++index) {
            if (index != 0) output << ',';
            output << '\"' << report.intervention_targets[index] << '\"';
        }
        output << ']'
               << ",\"intervention_validated\":true"
               << ",\"intervention_applied\":"
               << (report.intervention_applied ? "true" : "false");
        if (!report.intervention_metrics.empty()) {
            output << ",\"intervention_metrics\":[" << std::defaultfloat
                   << std::setprecision(12);
            for (std::size_t index = 0;
                 index < report.intervention_metrics.size(); ++index) {
                if (index != 0) output << ',';
                const InterventionMetric & metric = report.intervention_metrics[index];
                output << "{\"tensor_name\":\"" << metric.tensor_name << "\""
                       << ",\"source_projection_l2\":" << metric.source_projection_l2
                       << ",\"stored_projection_l2\":" << metric.stored_projection_l2
                       << ",\"stored_projection_ratio\":" << metric.stored_projection_ratio
                       << ",\"signed_projection_coefficient\":"
                       << metric.signed_projection_coefficient
                       << ",\"relative_frobenius_delta\":"
                       << metric.relative_frobenius_delta
                       << ",\"row_norm_relative_rmse\":"
                       << metric.row_norm_relative_rmse
                       << ",\"row_norm_relative_max\":"
                       << metric.row_norm_relative_max << '}';
            }
            output << ']';
        }
    }
    output << "}";
    return output.str();
}

}  // namespace ember::gguf_quantize
