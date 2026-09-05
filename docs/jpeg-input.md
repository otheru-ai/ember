# Native JPEG input

The native DeepSeek vision request path accepts 8-bit Huffman-coded baseline
(SOF0) and progressive (SOF2) JPEG, with grayscale or RGB/YCbCr components.
It produces RGB8 and uses the existing PNG resize, normalization, patch packing
and tower path. Format selection uses bytes, not an assumed MIME label.

CMYK/YCCK, arithmetic/lossless/hierarchical and non-8-bit JPEG are rejected;
WebP and GIF remain unsupported. EXIF orientation and ICC color transforms are
not applied. Normalize orientation/color profiles before upload when required.

Before libjpeg-turbo is created, a complete marker walk enforces one frame,
positive dimensions <=16384 per axis, <=16777216 pixels, <=64MiB encoded data,
valid segment bounds, bounded component sampling, <=64 scans and an EOI at the
exact end of the input. The HTTP body ceiling applies separately, including
base64 overhead. Duplicate/unknown scan components and nested images fail.
The marker walk is not an entropy decoder: malformed quantization/Huffman/scan
data is validated by libjpeg-turbo. All codec errors and warnings reject the
image, including partial-decode warnings. No partial RGB or patches escape.

The RGB output allocation is bounded by the pixel cap (48MiB). Codec work and
coefficient storage also depend on bounded dimensions, sampling and scans.
These are input/work bounds, not an OS memory sandbox or a hard whole-process
RSS/time limit. JPEG parsing runs in the existing request worker before loading
or locking the vision tower. libjpeg-turbo remains native request-facing code;
use maintained distribution packages and security updates.

Build dependency: libturbojpeg0-dev on Debian/Ubuntu (TurboJPEG >=2.0 APIs).
CI and Docker toolchain install it; release collect-runtime.sh copies the
linked library into the runtime closure, and the package copyright is retained.
No permissive fallback is compiled when that dependency is missing.

The existing ds4_vision_native_contract test includes retained Pillow RGB
fixtures for grayscale, 4:4:4, 4:2:2, 4:2:0 and progressive images; equivalent
PNG pixels must produce identical patches for all four prompt offsets. Fixtures
and hashes are in test/fixtures/jpeg/manifest.json. Pillow uses libjpeg-turbo too:
this checks wrapper configuration, pixels and pipeline equivalence, not an
independent entropy/DCT implementation. It is not a model-answer oracle.

Malformed checks include every truncated fixture prefix, wrong magic, trailing
bytes, concatenated images, invalid segment/component headers, zero/huge
dimensions, coding modes, a 65-scan stream, corrupt quantization, missing entropy
with a retained EOI (must fail inside the codec), and byte mutations checking
bounded success or cleared failure outputs. Sanitizers instrument Ember code;
the distribution codec binary is not rebuilt with instrumentation by that gate.

Before deployment, independently review the exact source and run the real JPEG
request through the candidate engine/model after the current fixed-N study
finishes. Compare against the same JPEG decoded to PNG with matching RGB pixels.
Host decoder/patch tests alone do not certify GPU/model behavior.
