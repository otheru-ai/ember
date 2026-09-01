// Learned language-boundary marker loader for DeepSeek-V4-Flash-Vision-Exp.
//
// The four marker rows are ordinary F32 tensors in the mmproj GGUF. They are
// deliberately loaded independently from the F16 tower and aligner weights:
// no dequantization or device execution is needed to establish the exact
// language-side image block. The caller supplies the text model's embedding
// width, and every marker must be one contiguous row of exactly that width.

#pragma once

#include "deepseek4_vision_contract.h"

#include <cstdint>
#include <string>

namespace dflash {

// Operator-supplied mmproj path only. Request media never supplies this path.
bool deepseek4_load_image_markers(const std::string & mmproj_path,
                                  int32_t model_n_embd,
                                  Deepseek4ImageMarkers & out,
                                  std::string & error);

}  // namespace dflash
