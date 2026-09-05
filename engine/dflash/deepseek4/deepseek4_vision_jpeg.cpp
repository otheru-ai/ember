#include "deepseek4_vision_native_contract.h"

#include <limits>
#include <turbojpeg.h>

namespace dflash {
namespace {
bool fail(std::string * error, const char * message) {
    if (error) *error = message;
    return false;
}
unsigned be16(const uint8_t * p) {
    return (static_cast<unsigned>(p[0]) << 8) | p[1];
}
struct Decoder {
    tjhandle handle = tjInitDecompress();
    ~Decoder() { if (handle) tjDestroy(handle); }
};
} // namespace

bool deepseek4_vision_validate_still_jpeg(
        const uint8_t * data, size_t size, int max_dimension,
        uint64_t max_pixels, Deepseek4VisionJpegInfo & out,
        std::string * error) {
    out = {};
    if (error) error->clear();
    if (!data || size < 4 ||
        size > deepseek4_vision_native_config().max_encoded_bytes ||
        max_dimension <= 0 || max_pixels == 0 ||
        data[0] != 0xff || data[1] != 0xd8)
        return fail(error, "invalid or oversized DeepSeek4 JPEG input");
    Deepseek4VisionJpegInfo info;
    uint8_t component_ids[3] = {};
    size_t pos = 2;
    bool frame = false;
    while (pos < size) {
        if (data[pos++] != 0xff)
            return fail(error, "JPEG data outside an entropy scan");
        while (pos < size && data[pos] == 0xff) ++pos;
        if (pos == size) return fail(error, "truncated JPEG marker");
        const unsigned marker = data[pos++];
        if (marker == 0xd9) {
            if (!frame || info.scans == 0 || pos != size)
                return fail(error, "JPEG requires one complete image and terminal EOI");
            out = info;
            return true;
        }
        // Allowed marker subset: Huffman DCT baseline/progressive and ordinary
        // tables/metadata. Reject arithmetic, lossless, hierarchical, DNL,
        // nested SOI and standalone restart markers before codec allocation.
        if (!(marker == 0xc0 || marker == 0xc2 || marker == 0xc4 ||
              marker == 0xdb || marker == 0xdd || marker == 0xda ||
              marker == 0xfe || (marker >= 0xe0 && marker <= 0xef)))
            return fail(error, "unsupported JPEG marker or coding mode");
        if (size - pos < 2) return fail(error, "truncated JPEG segment length");
        const size_t length = be16(data + pos);
        if (length < 2 || length > size - pos)
            return fail(error, "invalid JPEG segment length");
        const uint8_t * payload = data + pos + 2;
        const size_t payload_size = length - 2;
        if (marker == 0xc0 || marker == 0xc2) {
            if (frame || payload_size < 6 || payload[0] != 8 ||
                (payload[5] != 1 && payload[5] != 3) ||
                payload_size != 6u + 3u * payload[5])
                return fail(error, "JPEG requires one 8-bit grayscale or three-component frame");
            info.height = static_cast<int>(be16(payload + 1));
            info.width = static_cast<int>(be16(payload + 3));
            info.channels = payload[5];
            info.progressive = marker == 0xc2;
            const uint64_t pixels = static_cast<uint64_t>(info.width) *
                                    static_cast<uint64_t>(info.height);
            if (info.width <= 0 || info.height <= 0 ||
                info.width > max_dimension || info.height > max_dimension ||
                pixels > max_pixels || pixels > std::numeric_limits<size_t>::max() / 3u)
                return fail(error, "JPEG dimensions exceed decode limits");
            unsigned blocks = 0;
            for (int i = 0; i < info.channels; ++i) {
                const uint8_t * component = payload + 6 + 3 * i;
                const unsigned h = component[1] >> 4;
                const unsigned v = component[1] & 15u;
                if (h == 0 || h > 4 || v == 0 || v > 4 || component[2] > 3)
                    return fail(error, "invalid JPEG component sampling");
                for (int j = 0; j < i; ++j)
                    if (component_ids[j] == component[0])
                        return fail(error, "duplicate JPEG component identifier");
                component_ids[i] = component[0];
                blocks += h * v;
            }
            if (blocks > 10) return fail(error, "JPEG MCU exceeds sampling limit");
            frame = true;
        } else if (marker == 0xda) {
            if (!frame || payload_size < 1 || payload[0] == 0 ||
                payload[0] > info.channels || payload_size != 4u + 2u * payload[0])
                return fail(error, "invalid JPEG scan header");
            if (++info.scans > 64) return fail(error, "JPEG exceeds 64-scan limit");
            unsigned selected = 0;
            for (unsigned i = 0; i < payload[0]; ++i) {
                unsigned bit = 0;
                for (int j = 0; j < info.channels; ++j)
                    if (payload[1 + 2 * i] == component_ids[j])
                        bit = 1u << static_cast<unsigned>(j);
                if (bit == 0 || (selected & bit) != 0)
                    return fail(error, "invalid JPEG scan component identifier");
                selected |= bit;
            }
        }
        pos += length;
        if (marker == 0xda) {
            // Walk byte stuffing and restart markers, leaving the next actual
            // marker for the outer parser. The codec checks entropy validity.
            while (pos < size) {
                if (data[pos] != 0xff) { ++pos; continue; }
                const size_t start = pos++;
                while (pos < size && data[pos] == 0xff) ++pos;
                if (pos == size) return fail(error, "truncated JPEG entropy marker");
                const unsigned next = data[pos];
                if (next == 0 || (next >= 0xd0 && next <= 0xd7)) {
                    ++pos;
                    continue;
                }
                pos = start;
                break;
            }
        }
    }
    return fail(error, "JPEG is missing terminal EOI");
}

bool deepseek4_vision_decode_still_jpeg_rgb8(
        const uint8_t * data, size_t size, int max_dimension,
        uint64_t max_pixels, std::vector<uint8_t> & rgb,
        Deepseek4VisionJpegInfo & out, std::string * error) {
    rgb.clear();
    out = {};
    Deepseek4VisionJpegInfo info;
    if (!deepseek4_vision_validate_still_jpeg(
            data, size, max_dimension, max_pixels, info, error)) return false;
    Decoder decoder;
    if (!decoder.handle) return fail(error, "cannot initialize JPEG decoder");
    int width = 0, height = 0, subsampling = 0, colorspace = 0;
    if (size > std::numeric_limits<unsigned long>::max() ||
        tjDecompressHeader3(decoder.handle, data, static_cast<unsigned long>(size),
                            &width, &height, &subsampling, &colorspace) != 0 ||
        width != info.width || height != info.height ||
        (colorspace != TJCS_RGB && colorspace != TJCS_YCbCr && colorspace != TJCS_GRAY))
        return fail(error, "invalid or unsupported JPEG header");
    std::vector<uint8_t> decoded(static_cast<size_t>(width) *
                                 static_cast<size_t>(height) * 3u);
    // libjpeg-turbo API: STOPONWARNING rejects partial/corrupt decodes;
    // ACCURATEDCT selects the integer reference DCT (Pillow default).
    // LIMITSCANS is defense in depth; our complete preflight uses a tighter64.
    // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/2.1.5/turbojpeg.h
    if (tjDecompress2(decoder.handle, data, static_cast<unsigned long>(size),
                      decoded.data(), width, 0, height, TJPF_RGB,
                      TJFLAG_STOPONWARNING | TJFLAG_ACCURATEDCT | TJFLAG_LIMITSCANS) != 0)
        return fail(error, "JPEG decode rejected corrupt or incomplete image data");
    rgb.swap(decoded);
    out = info;
    return true;
}
} // namespace dflash
