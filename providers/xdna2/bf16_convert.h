#ifndef EMBER_PROVIDERS_XDNA2_BF16_CONVERT_H
#define EMBER_PROVIDERS_XDNA2_BF16_CONVERT_H

#include <cstdint>
#include <cstring>

namespace ember_xdna2 {

// Round-to-nearest-even float -> bfloat16.
//
// The exponent guard is load-bearing, not decoration. Without it the rounding
// addend is applied to NaN and Inf bit patterns and can carry out of the
// mantissa, through the exponent and into the sign bit: 0x7FFFFFFF (a NaN with
// every mantissa bit set) becomes 0x8000, negative zero. A NaN is the signal
// that a weight or activation is broken, and these conversions run in the
// quantization and validation paths whose job is to catch exactly that, so
// turning it into a plausible-looking zero is the worst available outcome.
//
// Finite values are unaffected by the guard, including FLT_MAX correctly
// rounding up to Inf.
inline uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline float bf16_round(float value) {
    return bf16_to_float(float_to_bf16(value));
}

}  // namespace ember_xdna2

#endif  // EMBER_PROVIDERS_XDNA2_BF16_CONVERT_H
