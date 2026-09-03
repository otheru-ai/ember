#include "deepseek4_vision_mmproj.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash;

static int g_pass;
static int g_fail;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++g_pass; }                                                \
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); }           \
} while (0)

namespace {

enum class Mutation {
    None,
    MissingTensor,
    UnknownTensor,
    WrongTensorType,
    WrongTensorShape,
    WrongProjector,
    WrongNormEpsilon,
    WrongScaleFactor,
};

struct Fixture {
    ggml_context * meta = nullptr;
    gguf_context * gguf = nullptr;

    explicit Fixture(Mutation mutation) {
        ggml_init_params params {
            /*.mem_size=*/16 * 1024 * 1024,
            /*.mem_buffer=*/nullptr,
            /*.no_alloc=*/true,
        };
        meta = ggml_init(params);
        gguf = gguf_init_empty();
        if (!meta || !gguf) std::abort();
        const auto & config = deepseek4_vision_native_config();
        gguf_set_val_str(gguf, "clip.projector_type",
                         mutation == Mutation::WrongProjector
                             ? "otherarch_merger" : "deepseekv4vision");
        gguf_set_val_f32(gguf,
            "clip.vision.attention.layer_norm_epsilon",
            mutation == Mutation::WrongNormEpsilon
                ? 1.0e-20f : config.norm_epsilon);
        gguf_set_val_bool(gguf, "clip.use_silu", true);
        gguf_set_val_u32(gguf, "clip.vision.projector.scale_factor",
            static_cast<uint32_t>(mutation == Mutation::WrongScaleFactor
                ? 2 : config.scale_factor));
        gguf_set_val_u32(gguf, "clip.vision.image_min_pixels",
                         static_cast<uint32_t>(config.image_min_pixels));
        gguf_set_val_f32(gguf, "clip.vision.rope_theta", config.rope_theta);
        gguf_set_val_u32(gguf, "clip.vision.max_n_token",
                         static_cast<uint32_t>(config.max_n_token));
        gguf_set_val_f32(gguf, "clip.vision.max_wh_ratio",
                         config.max_wh_ratio);
        gguf_set_val_u32(gguf, "clip.vision.image_size",
                         static_cast<uint32_t>(config.nominal_image_size));
        gguf_set_val_u32(gguf, "clip.vision.patch_size",
                         static_cast<uint32_t>(config.patch_size));
        gguf_set_val_u32(gguf, "clip.vision.embedding_length",
                         static_cast<uint32_t>(config.embedding_length));
        gguf_set_val_u32(gguf, "clip.vision.feed_forward_length",
                         static_cast<uint32_t>(config.feed_forward_length));
        gguf_set_val_u32(gguf, "clip.vision.attention.head_count",
                         static_cast<uint32_t>(config.head_count));
        gguf_set_val_u32(gguf, "clip.vision.block_count",
                         static_cast<uint32_t>(config.block_count));

        auto specs = deepseek4_vision_tensor_specs();
        for (size_t i = 0; i < specs.size(); ++i) {
            if (mutation == Mutation::MissingTensor && i + 1 == specs.size()) {
                continue;
            }
            auto spec = specs[i];
            if (mutation == Mutation::UnknownTensor && i + 1 == specs.size()) {
                spec.name = "v.unknown.weight";
            }
            ggml_type type = spec.storage == Deepseek4VisionStorage::F16
                ? GGML_TYPE_F16 : GGML_TYPE_F32;
            if (mutation == Mutation::WrongTensorType && i == 5) {
                type = type == GGML_TYPE_F16 ? GGML_TYPE_F32 : GGML_TYPE_F16;
            }
            std::vector<int64_t> shape = spec.shape;
            if (mutation == Mutation::WrongTensorShape && i == 5) {
                ++shape[0];
            }
            ggml_tensor * tensor = ggml_new_tensor(
                meta, type, static_cast<int>(shape.size()), shape.data());
            ggml_set_name(tensor, spec.name.c_str());
            gguf_add_tensor(gguf, tensor);
        }
    }

    ~Fixture() {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
    }
};

void test_exact_metadata_contract() {
    Fixture fixture(Mutation::None);
    Deepseek4VisionMmprojMetadata metadata;
    std::string error;
    CHECK(deepseek4_validate_vision_mmproj_metadata(
              fixture.gguf, fixture.meta, 4096, SIZE_MAX, metadata, error),
          "exact 299-tensor metadata fixture validates");
    CHECK(metadata.tensors.size() == 299 &&
              metadata.config.block_count == 32 &&
              metadata.config.norm_epsilon == 1.0e-6f,
          "validated metadata retains native tower configuration");
    CHECK(!deepseek4_validate_vision_mmproj_metadata(
              fixture.gguf, fixture.meta, 4095, SIZE_MAX, metadata, error),
          "language embedding-width mismatch fails closed");
    CHECK(!deepseek4_validate_vision_mmproj_metadata(
              fixture.gguf, fixture.meta, 4096, 0, metadata, error),
          "tensor slices extending past EOF fail closed");
}

void test_mutations_fail_closed() {
    const struct Case {
        Mutation mutation;
        const char * message;
    } cases[] = {
        {Mutation::MissingTensor, "missing tower tensor is rejected"},
        {Mutation::UnknownTensor, "unknown replacement tensor is rejected"},
        {Mutation::WrongTensorType, "wrong tensor storage type is rejected"},
        {Mutation::WrongTensorShape, "wrong GGML tensor orientation is rejected"},
        {Mutation::WrongProjector, "foreign projector architecture is rejected"},
        {Mutation::WrongNormEpsilon, "language-model epsilon is rejected for the tower"},
        {Mutation::WrongScaleFactor, "wrong pixel-shuffle factor is rejected"},
    };
    for (const auto & test : cases) {
        Fixture fixture(test.mutation);
        Deepseek4VisionMmprojMetadata metadata;
        std::string error;
        CHECK(!deepseek4_validate_vision_mmproj_metadata(
                  fixture.gguf, fixture.meta, 4096, SIZE_MAX,
                  metadata, error),
              test.message);
        CHECK(metadata.tensors.empty(),
              "rejected metadata leaves no partial tensor inventory");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    test_exact_metadata_contract();
    test_mutations_fail_closed();
    if (argc == 2) {
        Deepseek4VisionMmprojMetadata metadata;
        std::string error;
        if (!deepseek4_load_vision_mmproj_metadata(
                argv[1], 4096, metadata, error)) {
            std::fprintf(stderr, "real mmproj metadata load failed: %s\n",
                         error.c_str());
            return 1;
        }
        std::printf("real mmproj validated: %zu tensors\n",
                    metadata.tensors.size());
    }
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
