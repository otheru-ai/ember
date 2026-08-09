// Enumerate VK_KHR_cooperative_matrix configurations on the RADV device.
// g++ -O2 -o probe vk_coopmat_probe.cpp -lvulkan
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <vector>

static const char * ctype(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
        case VK_COMPONENT_TYPE_SINT8_KHR:   return "s8";
        case VK_COMPONENT_TYPE_SINT16_KHR:  return "s16";
        case VK_COMPONENT_TYPE_SINT32_KHR:  return "s32";
        case VK_COMPONENT_TYPE_UINT8_KHR:   return "u8";
        case VK_COMPONENT_TYPE_UINT16_KHR:  return "u16";
        case VK_COMPONENT_TYPE_UINT32_KHR:  return "u32";
        default: return "?";
    }
}

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        std::printf("vkCreateInstance failed\n"); return 1;
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    auto pfn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!pfn) { std::printf("coopmat entry point missing\n"); return 1; }

    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (std::strstr(p.deviceName, "llvmpipe")) continue;
        std::printf("device: %s\n", p.deviceName);
        std::printf("  maxComputeSharedMemorySize = %u B\n",
                    p.limits.maxComputeSharedMemorySize);
        std::printf("  maxComputeWorkGroupInvocations = %u\n",
                    p.limits.maxComputeWorkGroupInvocations);

        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(d, &p2);
        std::printf("  subgroupSize = %u\n", sg.subgroupSize);

        uint32_t cn = 0;
        pfn(d, &cn, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> props(
            cn, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        pfn(d, &cn, props.data());
        std::printf("  cooperative matrix configurations: %u\n", cn);
        for (auto & c : props) {
            std::printf("    M%-3u N%-3u K%-3u  A=%-4s B=%-4s C=%-4s D=%-4s  sat=%d scope=%u\n",
                        c.MSize, c.NSize, c.KSize, ctype(c.AType), ctype(c.BType),
                        ctype(c.CType), ctype(c.ResultType),
                        (int) c.saturatingAccumulation, (unsigned) c.scope);
        }
    }
    return 0;
}
