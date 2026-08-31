// Versioned offline aligner-output artifact for DeepSeek-V4-Flash-Vision-Exp.
//
// The artifact carries only row-major IMAGE embeddings and the required
// language-grid dimensions. Learned START/PAD/NEWLINE/END rows remain model
// weights in the mmproj and are joined through deepseek4_prepare_image(). This
// keeps offline validation and the later native tower on one language seam.
//
// Little-endian format (32-byte header):
//   char[8]  magic       "DS4VIMG\0"
//   uint32   version     2
//   int32    n_embd
//   int32    n_llm_h     required, never inferred from a flat row count
//   int32    n_llm_w     required, never inferred from a flat row count
//   uint32   flags       0
//   uint32   reserved    0
//   float32  image_rows[n_llm_h*n_llm_w*n_embd]

#ifndef DFLASH_DEEPSEEK4_VISION_ARTIFACT_H
#define DFLASH_DEEPSEEK4_VISION_ARTIFACT_H

#include <cstdint>
#include <string>
#include <vector>

namespace dflash {

struct Deepseek4VisionArtifact {
    int32_t n_embd = 0;
    int32_t n_llm_h = 0;
    int32_t n_llm_w = 0;
    std::vector<float> image_embeddings;
    uint64_t digest = 0;
};

bool deepseek4_parse_vision_artifact(
    const std::vector<uint8_t> & bytes, int32_t model_n_embd,
    Deepseek4VisionArtifact & out, std::string & error);

}  // namespace dflash

#endif  // DFLASH_DEEPSEEK4_VISION_ARTIFACT_H
