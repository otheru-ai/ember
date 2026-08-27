// Architecture-neutral contracts shared by Ember's streaming GGUF quantizer
// and its GPU-free tests. This file owns only CLI and tensor-policy decisions;
// GGUF I/O and numeric conversion remain in the engine-only executable.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ember::gguf_quantize {

enum class TensorFormat {
    rocmi4,
    q6_k,
    rocmfp4_fast,
};

struct TensorTypeOverride {
    std::string pattern;
    TensorFormat format = TensorFormat::rocmi4;
};

struct Options {
    bool build_info_json = false;
    bool dry_size_json = false;
    bool keep_split = false;
    bool budget_configured = false;
    std::uint64_t device_budget_bytes = 0;
    std::uint64_t runtime_reserve_bytes = 0;
    std::vector<TensorTypeOverride> tensor_type_overrides;
    std::string intervention_manifest_path;
    std::string input_path;
    std::string output_path;
    std::size_t threads = 0;
};

const char * tensor_format_name(TensorFormat format);

// Returns false with a user-facing error on malformed or unsupported syntax.
// Options are deliberately accepted only before the first positional.
bool parse_options(int argc, const char * const * argv,
                   Options & options, std::string & error);

// The ordinary architecture-neutral policy follows llama.cpp's conservative
// weight rule. A matching manual override is handled by the caller before this
// policy and therefore can force the PLE row-gather table to ROCMI4.
bool tensor_allows_quantization(const std::string & name, std::size_t rank);

std::string build_info_json(const std::string & ember_revision);

struct InterventionMetric {
    std::string tensor_name;
    double source_projection_l2 = 0.0;
    double stored_projection_l2 = 0.0;
    double stored_projection_ratio = 0.0;
    double signed_projection_coefficient = 0.0;
    double relative_frobenius_delta = 0.0;
    double row_norm_relative_rmse = 0.0;
    double row_norm_relative_max = 0.0;
};

struct SizeReport {
    std::uint64_t artifact_bytes = 0;
    std::vector<std::uint64_t> shard_bytes;
    std::uint64_t runtime_reserve_bytes = 0;
    std::uint64_t budget_bytes = 0;
    bool fits = false;
    std::string intervention_manifest_sha256;
    std::string intervention_target_names_sha256;
    std::vector<std::string> intervention_targets;
    std::vector<InterventionMetric> intervention_metrics;
    bool intervention_validated = false;
    bool intervention_applied = false;
};

std::string size_report_json(const SizeReport & report);

}  // namespace ember::gguf_quantize
