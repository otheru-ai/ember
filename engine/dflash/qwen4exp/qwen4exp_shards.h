// Ordered GGUF shard-set discovery for Qwen4Exp.
//
// This layer intentionally has no ggml or HIP dependency. Keeping filename,
// completeness, and inode-alias checks standalone gives the ordinary host test
// gauntlet coverage; semantic split metadata remains in qwen4exp_model.cpp.

#pragma once

#include <string>
#include <vector>

namespace dflash::common {

bool qwen4exp_discover_gguf_shards(const char * model_path,
                                  std::vector<std::string> & paths,
                                  std::string & error);

} // namespace dflash::common
