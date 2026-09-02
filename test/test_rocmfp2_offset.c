// Numerics model for ROCMFP2_OFFSET_FROM_DS (vecdotq.cuh,
// vec_dot_rocmfpx_fp2_q8_1): the affine offset term taken from
// block_q8_1.ds.y instead of a second dp4a over the activations.
//
// This is not a bit-identity test -- the two forms differ by design. It pins
// the size of that difference so the behavioural gate on the box has a
// pre-registered expectation: per 32-block, |exact - approx| must stay within
// offset * (18*d + |sum x| * 2^-11), from the integer-vs-float sum (16d), the
// f16 rounding of d carried into the exact form (~2d), and the f16 rounding of
// the stored sum. It also models quantize_q8_1 verbatim (amax/127, roundf,
// ds = half2(d, sum)) so a change there shows up here.
//
// Host test, no GPU. Requires _Float16 for the f16 rounding.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define QK8_1 32

static float round_f16(float x) {
    volatile _Float16 h = (_Float16) x;
    return (float) h;
}

// xorshift, fixed seed: the test is deterministic.
static uint32_t g_rng = 0x9E3779B9u;
static float frand(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return (float) (g_rng & 0xFFFFFFu) / (float) 0xFFFFFFu * 2.0f - 1.0f;
}

int main(void) {
    int fails = 0;
    double worst_ratio = 0.0;   // |delta| / bound
    double worst_rel   = 0.0;   // |delta| / (|scale term| + |offset term|), cancellation excluded
    const int nblocks = 200000;

    for (int b = 0; b < nblocks; ++b) {
        // Activation block, occasionally scaled to exercise wide magnitudes.
        float x[QK8_1];
        const float mag = (b % 7 == 0) ? 64.0f : (b % 5 == 0) ? 0.01f : 1.0f;
        float amax = 0.0f, sum = 0.0f;
        for (int i = 0; i < QK8_1; ++i) {
            x[i] = frand() * mag;
            if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
            sum += x[i];
        }
        const float d = amax / 127.0f;
        int8_t q[QK8_1];
        int sumq = 0;
        for (int i = 0; i < QK8_1; ++i) {
            q[i] = (int8_t) (amax == 0.0f ? 0 : (int) roundf(x[i] / d));
            sumq += q[i];
        }
        const float ds_x = round_f16(d);
        const float ds_y = round_f16(sum);

        // Weight block: 2-bit codes 0..3, UE4M3-shaped scale/offset magnitudes.
        int codes[QK8_1];
        for (int i = 0; i < QK8_1; ++i) codes[i] = (int) (frand() * 1.999f + 2.0f) & 3;
        const float scale  = 0.0625f * (1.0f + fabsf(frand()));
        const float offset = scale * (0.5f + fabsf(frand()) * 2.0f);

        int sumi = 0;
        for (int i = 0; i < QK8_1; ++i) sumi += codes[i] * q[i];

        // Both halves summed, as mmvq.cu reduces them.
        const double exact  = (double) ds_x * ((double) scale * sumi - (double) offset * sumq);
        const double approx = (double) ds_x * scale * sumi - (double) offset * ds_y;
        const double delta  = fabs(exact - approx);
        const double bound  = (double) offset * (18.0 * d + fabs(sum) * (1.0 / 2048.0)) + 1e-9;

        if (delta > bound) {
            if (fails < 8) {
                printf("FAIL: block %d delta=%.6g bound=%.6g d=%.6g sum=%.6g offset=%.6g\n",
                       b, delta, bound, d, sum, offset);
            }
            fails++;
        }
        if (delta / bound > worst_ratio) worst_ratio = delta / bound;
        const double terms = fabs((double) ds_x * scale * sumi) + fabs((double) offset * ds_y);
        if (terms > 1e-6 && delta / terms > worst_rel) worst_rel = delta / terms;
    }

    printf("%s: ds.y offset term within bound on %d blocks (worst delta/bound %.3f, worst rel %.3g)\n",
           fails ? "FAIL" : "PASS", nblocks, worst_ratio, worst_rel);
    return fails != 0;
}
