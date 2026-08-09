// Vulkan host harness for the coopmat MLA decode shader.
// g++ -O2 -o vk_bench vk_bench.cpp -lvulkan
#include <vulkan/vulkan.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#define D 512
#define NHEAD 64

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    std::fprintf(stderr, "%s:%d vk error %d\n", __FILE__, __LINE__, (int) r_); \
    std::exit(1); } } while (0)

static uint16_t f2h(float f) {          // round-to-nearest-even, matches __float2half
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    uint16_t h = (uint16_t)(s | (uint32_t) (e << 10) | (m >> 13));
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1))) h++;
    return h;
}
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu;
    if (e == 0) { if (!m) { float f; uint32_t b = s; std::memcpy(&f,&b,4); return f; }
        while (!(m & 0x400u)) { m <<= 1; e--; } e++; m &= 0x3ffu; }
    else if (e == 31) { uint32_t b = s | 0x7f800000u | (m << 13); float f;
        std::memcpy(&f,&b,4); return f; }
    uint32_t b = s | ((e + 112u) << 23) | (m << 13);
    float f; std::memcpy(&f, &b, 4); return f;
}

struct Buf { VkBuffer buf; VkDeviceMemory mem; void * ptr; size_t size; };

static VkDevice dev; static VkPhysicalDevice phys; static uint32_t qfam;
static VkQueue queue; static VkCommandPool cpool;

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    std::fprintf(stderr, "no memory type\n"); std::exit(1);
}

static Buf mkbuf(size_t size) {
    Buf b{}; b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memtype(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
    VK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    VK(vkMapMemory(dev, b.mem, 0, size, 0, &b.ptr));
    return b;
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 100;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst; VK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    phys = VK_NULL_HANDLE;
    for (auto d : pds) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
        if (!std::strstr(p.deviceName, "llvmpipe")) { phys = d; std::printf("device: %s\n", p.deviceName); break; } }
    if (!phys) { std::printf("no hw device\n"); return 1; }
    VkPhysicalDeviceProperties pprops; vkGetPhysicalDeviceProperties(phys, &pprops);
    const double ts_period = pprops.limits.timestampPeriod;

    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qf.data());
    qfam = 0; for (uint32_t i = 0; i < qn; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cm.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.shaderFloat16 = VK_TRUE; v12.storageBuffer8BitAccess = VK_FALSE;
    v12.vulkanMemoryModel = VK_TRUE; v12.vulkanMemoryModelDeviceScope = VK_TRUE;
    v12.pNext = &cm;
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11.storageBuffer16BitAccess = VK_TRUE; v11.pNext = &v12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &v11;

    const char * exts[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = exts;
    VK(vkCreateDevice(phys, &dci, nullptr, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qfam;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK(vkCreateCommandPool(dev, &cpi, nullptr, &cpool));

    // shader
    FILE * fp = std::fopen("mla.spv", "rb");
    if (!fp) { std::printf("mla.spv missing\n"); return 1; }
    std::fseek(fp, 0, SEEK_END); long sz = std::ftell(fp); std::fseek(fp, 0, SEEK_SET);
    std::vector<char> code(sz); if (std::fread(code.data(), 1, sz, fp) != (size_t) sz) return 1;
    std::fclose(fp);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sz; smi.pCode = (const uint32_t *) code.data();
    VkShaderModule sm; VK(vkCreateShaderModule(dev, &smi, nullptr, &sm));

    VkDescriptorSetLayoutBinding lb[6]{};
    for (int i = 0; i < 6; ++i) { lb[i].binding = i; lb[i].descriptorCount = 1;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 6; dli.pBindings = lb;
    VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout plo; VK(vkCreatePipelineLayout(dev, &pli, nullptr, &plo));
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm;
    cpci.stage.pName = "main"; cpci.layout = plo;
    VkPipeline pipe; VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    const int MAXSPLIT = 64;
    const int refs[] = {128, 256, 416, 512, 768, 960, 1616, 3278, 8896};
    const double hip_us[] = {14.6, 17.9, 21.6, 21.6, 24.9, 28.3, 38.1, 58.1, 124.6};
    int max_kv = 8896;

    Buf bq = mkbuf((size_t) NHEAD * D * 2);
    Buf bk = mkbuf((size_t) max_kv * D * 2);
    Buf bm = mkbuf((size_t) max_kv * 2);
    Buf ba = mkbuf((size_t) MAXSPLIT * NHEAD * D * 4);
    Buf bmm = mkbuf((size_t) MAXSPLIT * NHEAD * 4);
    Buf bll = mkbuf((size_t) MAXSPLIT * NHEAD * 4);

    unsigned seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u;
                       return ((float)(seed >> 8) / (float)(1 << 24) - 0.5f) * 2.0f; };
    std::vector<float> qvals((size_t) NHEAD * D);
    for (size_t i = 0; i < qvals.size(); ++i) { qvals[i] = rnd() * 0.5f; ((uint16_t *) bq.ptr)[i] = f2h(qvals[i]); }
    std::vector<float> kf((size_t) max_kv * D);
    for (size_t i = 0; i < kf.size(); ++i) { kf[i] = rnd() * 0.5f; ((uint16_t *) bk.ptr)[i] = f2h(kf[i]); }

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &dps;
    VkDescriptorPool dpool; VK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = dpool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
    VkDescriptorSet ds; VK(vkAllocateDescriptorSets(dev, &dsa, &ds));
    VkBuffer bufs[6] = {bq.buf, bk.buf, bm.buf, ba.buf, bmm.buf, bll.buf};
    size_t szs[6] = {bq.size, bk.size, bm.size, ba.size, bmm.size, bll.size};
    VkDescriptorBufferInfo dbi[6]; VkWriteDescriptorSet w[6]{};
    for (int i = 0; i < 6; ++i) {
        dbi[i] = {bufs[i], 0, szs[i]};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ds;
        w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 6, w, 0, nullptr);

    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
    VkQueryPool qpool; VK(vkCreateQueryPool(dev, &qpi, nullptr, &qpool));

    VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cba.commandPool = cpool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
    VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev, &cba, &cb));

    std::printf("%-8s %-6s %10s %10s %9s %10s\n", "n_kv", "split", "vulkan_us", "hip_wmma", "vk/hip", "maxerr");
    for (int ri = 0; ri < 9; ++ri) {
        const int n_kv = refs[ri];
        for (int r = 0; r < n_kv; ++r)
            ((uint16_t *) bm.ptr)[r] = f2h((r >= 120 && r < 128) ? -1.0e30f : 0.0f);
        const float scale = 1.0f / std::sqrt((float) D);

        double best = 1e30; int bsplit = 1;
        for (int split = 1; split <= MAXSPLIT; split *= 2) {
            if ((n_kv + split - 1) / split < 16) break;
            struct { int a, b; float c; } pcv{n_kv, split, scale};
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK(vkBeginCommandBuffer(cb, &bi));
            vkCmdResetQueryPool(cb, qpool, 0, 2);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plo, 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cb, plo, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pcv);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
            for (int i = 0; i < iters; ++i) {
                vkCmdDispatch(cb, split, 1, 1);
                VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
            VK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
            VK(vkQueueWaitIdle(queue));
            uint64_t ts[2];
            VK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const double us = (double)(ts[1] - ts[0]) * ts_period / 1000.0 / iters;
            if (us < best) { best = us; bsplit = split; }
        }
        // correctness: re-dispatch once at the winning split, combine the
        // partials on the host, compare against a double-precision reference
        // over exactly the f16 values the shader consumed.
        double maxerr = -1.0;
        if (n_kv <= 1024) {
            struct { int a, b; float c; } pcv{n_kv, bsplit, scale};
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK(vkBeginCommandBuffer(cb, &bi));
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plo, 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cb, plo, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pcv);
            vkCmdDispatch(cb, bsplit, 1, 1);
            VK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
            VK(vkQueueWaitIdle(queue));

            const float * pa = (const float *) ba.ptr;
            const float * pm = (const float *) bmm.ptr;
            const float * pl = (const float *) bll.ptr;
            std::vector<double> ref((size_t) NHEAD * D, 0.0), pr(n_kv);
            for (int h = 0; h < NHEAD; ++h) {
                double m = -INFINITY;
                for (int r = 0; r < n_kv; ++r) {
                    const float mv = h2f(((uint16_t *) bm.ptr)[r]);
                    if (mv <= -1.0e20f) { pr[r] = -INFINITY; continue; }
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d)
                        dot += (double) h2f(((uint16_t *) bq.ptr)[(size_t) h * D + d]) *
                               (double) h2f(((uint16_t *) bk.ptr)[(size_t) r * D + d]);
                    pr[r] = dot * scale + mv;
                    if (pr[r] > m) m = pr[r];
                }
                double den = 0.0;
                for (int r = 0; r < n_kv; ++r) {
                    pr[r] = pr[r] == -INFINITY ? 0.0 : std::exp(pr[r] - m);
                    den += pr[r];
                }
                for (int r = 0; r < n_kv; ++r) {
                    if (pr[r] == 0.0) continue;
                    const double w = pr[r] / den;
                    for (int d = 0; d < D; ++d)
                        ref[(size_t) h * D + d] +=
                            w * (double) h2f(((uint16_t *) bk.ptr)[(size_t) r * D + d]);
                }
                // combine this head's split partials (no sink, matching the ref)
                double gm = -INFINITY;
                for (int sp = 0; sp < bsplit; ++sp) gm = std::max(gm, (double) pm[sp * NHEAD + h]);
                double gl = 0.0;
                for (int sp = 0; sp < bsplit; ++sp) {
                    const double ms = pm[sp * NHEAD + h];
                    if (ms == (double) -3.402823466e38) continue;
                    gl += pl[sp * NHEAD + h] * std::exp(ms - gm);
                }
                for (int d = 0; d < D; ++d) {
                    double v = 0.0;
                    for (int sp = 0; sp < bsplit; ++sp) {
                        const double ms = pm[sp * NHEAD + h];
                        if (ms == (double) -3.402823466e38) continue;
                        v += pa[((size_t) sp * NHEAD + h) * D + d] * std::exp(ms - gm);
                    }
                    v = gl > 0.0 ? v / gl : 0.0;
                    maxerr = std::max(maxerr, std::fabs(v - ref[(size_t) h * D + d]));
                }
            }
        }
        std::printf("%-8d %-6d %10.1f %10.1f %8.2fx %10.2g\n", n_kv, bsplit, best,
                    hip_us[ri], hip_us[ri] > 0 ? best / hip_us[ri] : 0.0, maxerr);
    }
    return 0;
}
