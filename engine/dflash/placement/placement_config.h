// Fixed local-device configuration for Ember's single gfx1151 GPU.

#pragma once

namespace dflash::common {

struct DevicePlacement {
    int gpu     = 0;
    int max_ctx = 8192;
};

}  // namespace dflash::common
