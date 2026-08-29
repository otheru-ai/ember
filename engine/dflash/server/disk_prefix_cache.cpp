// Disk-backed prefix cache implementation.

#include "disk_prefix_cache.h"
#include "common/sha1.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>
#include <fcntl.h>
#include <unistd.h>

namespace dflash::common {

namespace fs = std::filesystem;

// ─── Utility ────────────────────────────────────────────────────────────

static std::string hex(const uint8_t * data, int len) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (int i = 0; i < len; ++i) {
        out.push_back(hex_chars[data[i] >> 4]);
        out.push_back(hex_chars[data[i] & 0x0f]);
    }
    return out;
}

static bool mkdir_p(const std::string & path) {
    std::error_code ec;
    const fs::file_status before = fs::symlink_status(path, ec);
    if (!ec && fs::exists(before) &&
        (fs::is_symlink(before) || !fs::is_directory(before))) {
        return false;
    }
    ec.clear();
    if (!fs::exists(before)) fs::create_directories(path, ec);
    if (ec || !fs::is_directory(fs::symlink_status(path, ec)) || ec) {
        return false;
    }
    // Cache snapshots may contain prompt-derived model state. Keep both the
    // namespace and predictable temporary names private to the serving uid.
    fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace, ec);
    return !ec;
}

static FILE * open_read_no_follow(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return nullptr;
    FILE * f = ::fdopen(fd, "rb");
    if (!f) ::close(fd);
    return f;
}

static FILE * open_write_no_follow(const std::string & path) {
    const int fd = ::open(path.c_str(),
                          O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW,
                          0600);
    if (fd < 0) return nullptr;
    FILE * f = ::fdopen(fd, "wb");
    if (!f) ::close(fd);
    return f;
}

static bool sync_directory(const std::string & path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

static uint64_t now_unix() {
    return (uint64_t)std::time(nullptr);
}

static bool disk_token_count_fits(size_t count) {
    return count <= (size_t)INT_MAX && count <= (size_t)UINT32_MAX;
}

static void add_total_bytes_saturating(uint64_t & total, uint64_t amount) {
    total = amount > UINT64_MAX - total ? UINT64_MAX : total + amount;
}

static PrefixHash hash_prefix(const int32_t * ids, int count) {
    if (count < 0 || (count > 0 && !ids) ||
        (size_t)count > (std::numeric_limits<size_t>::max() - 4) /
                            sizeof(int32_t)) {
        throw std::invalid_argument("invalid prefix token span");
    }

    std::vector<uint8_t> bytes(4 + (size_t)count * sizeof(int32_t));
    const uint32_t n = static_cast<uint32_t>(count);
    for (unsigned b = 0; b < 4; ++b) {
        bytes[b] = static_cast<uint8_t>(n >> (8U * b));
    }
    for (int i = 0; i < count; ++i) {
        const uint32_t token = static_cast<uint32_t>(ids[i]);
        const size_t off = 4 + static_cast<size_t>(i) * 4;
        for (unsigned b = 0; b < 4; ++b) {
            bytes[off + b] = static_cast<uint8_t>(token >> (8U * b));
        }
    }

    uint8_t sha[20];
    sha1_hash(bytes.data(), bytes.size(), sha);
    PrefixHash hash{};
    std::memcpy(hash.data(), sha, hash.size());
    return hash;
}

// #6: Shared CPU backend for disk-cache loads. A fresh ggml CPU backend used to
// be created (and intentionally leaked) on every disk-cache hit, so repeated
// hits leaked backends without bound. The CPU backend is stateless and
// CPU-allocated buffers reference the global CPU buffer type rather than this
// backend object, so a single process-wide instance can be reused across all
// loads and safely outlives every buffer allocated from it. The function-local
// static holder frees it exactly once at process exit. (DiskPrefixCache's
// destructor is defaulted in the header, which is owned elsewhere, so the
// backend cannot be stored as a per-instance member.)
static ggml_backend_t disk_cache_cpu_backend() {
    struct CpuBackendHolder {
        ggml_backend_t backend = ggml_backend_cpu_init();
        ~CpuBackendHolder() {
            if (backend) {
                ggml_backend_free(backend);
                backend = nullptr;
            }
        }
    };
    static CpuBackendHolder holder;  // thread-safe init; freed at process exit
    return holder.backend;
}

// Little-endian I/O helpers.
static void write_u32(FILE * f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static void write_u64(FILE * f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
static void write_i64(FILE * f, int64_t v)  { std::fwrite(&v, 8, 1, f); }
static void write_u16(FILE * f, uint16_t v) { std::fwrite(&v, 2, 1, f); }
static void write_u8(FILE * f, uint8_t v)   { std::fwrite(&v, 1, 1, f); }

static bool read_u32(FILE * f, uint32_t & out) { return std::fread(&out, 4, 1, f) == 1; }
static bool read_u64(FILE * f, uint64_t & out) { return std::fread(&out, 8, 1, f) == 1; }
static bool read_i64(FILE * f, int64_t & out)  { return std::fread(&out, 8, 1, f) == 1; }
static bool read_u16(FILE * f, uint16_t & out) { return std::fread(&out, 2, 1, f) == 1; }
static bool read_u8(FILE * f, uint8_t & out)   { return std::fread(&out, 1, 1, f) == 1; }

struct DiskTensorPatch {
    DiskTensorEntry tensor;
    uint64_t patch_offset = 0;
    uint64_t patch_bytes = 0;
};

struct ParsedDiskFile {
    DiskCacheHeader hdr{};
    PrefixHash parent_hash{};
    uint32_t parent_tokens = 0;
    uint32_t chain_depth = 0;
    std::vector<DiskTensorPatch> tensors;
    long payload_offset = 0;
};

static bool hash_is_zero(const PrefixHash & hash) {
    for (uint8_t b : hash) if (b != 0) return false;
    return true;
}

static bool snapshot_tensor_can_grow(const std::string & name) {
    return name.rfind("ds4_comp_kv_", 0) == 0 ||
           name.rfind("ds4_index_comp_kv_", 0) == 0;
}

static bool tensor_shape_prefix(const DiskTensorEntry & small,
                                const DiskTensorEntry & large) {
    if (small.name != large.name || small.type != large.type) return false;
    for (int d = 0; d < 4; ++d) {
        if (d == 1) {
            if (small.ne[d] > large.ne[d]) return false;
        } else if (small.ne[d] != large.ne[d]) {
            return false;
        }
    }
    return small.nbytes <= large.nbytes;
}

static bool parse_disk_file(const std::string & path, ParsedDiskFile & out) {
    out = {};
    FILE * f = open_read_no_follow(path);
    if (!f) return false;

    auto fail = [&]() {
        std::fclose(f);
        return false;
    };

    // Header is intentionally decoded here rather than through the private
    // class helper so recursive parent materialization remains self-contained.
    if (std::fread(out.hdr.magic, 1, 4, f) != 4 ||
        !read_u32(f, out.hdr.version) ||
        std::fread(out.hdr.layout_id, 1, 16, f) != 16 ||
        !read_u32(f, out.hdr.cur_pos) ||
        !read_u32(f, out.hdr.n_tensors) ||
        !read_u32(f, out.hdr.token_count) ||
        std::fread(out.hdr.token_hash, 1, 16, f) != 16 ||
        !read_u64(f, out.hdr.payload_bytes) ||
        !read_u64(f, out.hdr.created_at) ||
        !read_u64(f, out.hdr.last_used)) {
        return fail();
    }
    uint32_t last_tok = 0;
    if (!read_u32(f, last_tok)) return fail();
    out.hdr.last_tok = (int32_t)last_tok;

    if (std::memcmp(out.hdr.magic, "DKVC", 4) != 0 ||
        (out.hdr.version != DISK_CACHE_VERSION_V1 &&
         out.hdr.version != DISK_CACHE_VERSION_V2 &&
         out.hdr.version != DISK_CACHE_VERSION) ||
        out.hdr.n_tensors == 0 || out.hdr.n_tensors > 4096 ||
        out.hdr.cur_pos != out.hdr.token_count) {
        return fail();
    }

    // v3+ reason block, between the 80-byte base header and the v2 parent
    // descriptor. v1/v2 have no block and stay reason=UNKNOWN (0).
    if (out.hdr.version >= DISK_CACHE_VERSION) {
        uint8_t reason = 0, ext = 0;
        uint16_t rsv = 0;
        if (!read_u8(f, reason) || !read_u8(f, ext) || !read_u16(f, rsv)) {
            return fail();
        }
        out.hdr.reason = reason;
        out.hdr.ext_flags = ext;
    }

    // v2+ delta-chain parent descriptor.
    if (out.hdr.version >= DISK_CACHE_VERSION_V2) {
        if (std::fread(out.parent_hash.data(), 1, out.parent_hash.size(), f) !=
                out.parent_hash.size() ||
            !read_u32(f, out.parent_tokens) ||
            !read_u32(f, out.chain_depth) ||
            out.chain_depth > DISK_CACHE_MAX_CHAIN_DEPTH) {
            return fail();
        }
        if ((hash_is_zero(out.parent_hash) && out.chain_depth != 0) ||
            (!hash_is_zero(out.parent_hash) &&
             (out.chain_depth == 0 ||
              out.parent_tokens >= out.hdr.token_count))) {
            return fail();
        }
    }

    out.tensors.clear();
    out.tensors.reserve(out.hdr.n_tensors);
    uint64_t patch_total = 0;
    for (uint32_t i = 0; i < out.hdr.n_tensors; ++i) {
        DiskTensorPatch patch;
        uint16_t name_len = 0;
        if (!read_u16(f, name_len) || name_len == 0 ||
            name_len >= GGML_MAX_NAME) {
            return fail();
        }
        char name[GGML_MAX_NAME] = {};
        if (std::fread(name, 1, name_len, f) != name_len) return fail();
        patch.tensor.name = name;
        if (std::any_of(out.tensors.begin(), out.tensors.end(),
                        [&](const DiskTensorPatch & existing) {
                            return existing.tensor.name == patch.tensor.name;
                        }) ||
            !read_u32(f, patch.tensor.type) ||
            patch.tensor.type >= GGML_TYPE_COUNT) {
            return fail();
        }
        for (int d = 0; d < 4; ++d) {
            if (!read_i64(f, patch.tensor.ne[d]) ||
                patch.tensor.ne[d] <= 0) {
                return fail();
            }
        }
        uint64_t nbytes = 0;
        if (!read_u64(f, nbytes) || nbytes > SIZE_MAX) return fail();
        patch.tensor.nbytes = (size_t)nbytes;
        // The per-tensor patch descriptor (offset/bytes) is a v2+ feature; both
        // v2 and v3 carry it. v1 stores whole tensors (offset 0, full nbytes).
        if (out.hdr.version >= DISK_CACHE_VERSION_V2) {
            if (!read_u64(f, patch.patch_offset) ||
                !read_u64(f, patch.patch_bytes)) {
                return fail();
            }
        } else {
            patch.patch_offset = 0;
            patch.patch_bytes = nbytes;
        }
        if (patch.patch_offset > nbytes ||
            patch.patch_bytes > nbytes - patch.patch_offset ||
            patch_total > UINT64_MAX - patch.patch_bytes) {
            return fail();
        }
        patch_total += patch.patch_bytes;
        out.tensors.push_back(std::move(patch));
    }
    if (patch_total != out.hdr.payload_bytes) return fail();
    out.payload_offset = std::ftell(f);
    if (out.payload_offset < 0 ||
        out.hdr.payload_bytes >
            (uint64_t)LONG_MAX - (uint64_t)out.payload_offset ||
        std::fseek(f, 0, SEEK_END) != 0) {
        return fail();
    }
    const long file_size = std::ftell(f);
    if (file_size < 0 ||
        (uint64_t)file_size !=
            (uint64_t)out.payload_offset + out.hdr.payload_bytes) {
        return fail();
    }
    std::fclose(f);
    return true;
}

static bool read_file_tensor_payload(const std::string & path,
                                     const ParsedDiskFile & parsed,
                                     const char * name,
                                     std::vector<uint8_t> & out) {
    uint64_t offset = (uint64_t)parsed.payload_offset;
    const DiskTensorPatch * found = nullptr;
    for (const auto & patch : parsed.tensors) {
        if (patch.tensor.name == name) {
            found = &patch;
            break;
        }
        if (offset > UINT64_MAX - patch.patch_bytes) return false;
        offset += patch.patch_bytes;
    }
    if (!found || found->patch_offset != 0 ||
        found->patch_bytes != found->tensor.nbytes ||
        offset > (uint64_t)LONG_MAX ||
        found->patch_bytes > SIZE_MAX) {
        return false;
    }

    FILE * f = open_read_no_follow(path);
    if (!f) return false;
    const bool seek_ok = std::fseek(f, (long)offset, SEEK_SET) == 0;
    if (!seek_ok) {
        std::fclose(f);
        return false;
    }
    out.resize((size_t)found->patch_bytes);
    const bool read_ok =
        out.empty() ||
        std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return read_ok;
}

static bool read_file_snapshot_meta(const std::string & path,
                                    const ParsedDiskFile & parsed,
                                    std::vector<int32_t> & out) {
    std::vector<uint8_t> bytes;
    if (!read_file_tensor_payload(path, parsed, "ds4_snap_meta", bytes) ||
        bytes.empty() || bytes.size() % sizeof(int32_t) != 0) {
        return false;
    }
    out.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

static bool read_context_snapshot_meta(ggml_context * ctx,
                                       std::vector<int32_t> & out) {
    ggml_tensor * meta = ggml_get_tensor(ctx, "ds4_snap_meta");
    if (!meta || meta->type != GGML_TYPE_I32 ||
        ggml_nbytes(meta) == 0 ||
        ggml_nbytes(meta) % sizeof(int32_t) != 0) {
        return false;
    }
    out.resize(ggml_nbytes(meta) / sizeof(int32_t));
    ggml_backend_tensor_get(meta, out.data(), 0, ggml_nbytes(meta));
    return true;
}

static int compressed_counter_index(const std::string & name) {
    int layer = -1;
    int consumed = 0;
    if (std::sscanf(name.c_str(), "ds4_comp_kv_%d%n",
                    &layer, &consumed) == 1 &&
        consumed == (int)name.size() && layer >= 0) {
        return layer * 2;
    }
    layer = -1;
    consumed = 0;
    if (std::sscanf(name.c_str(), "ds4_index_comp_kv_%d%n",
                    &layer, &consumed) == 1 &&
        consumed == (int)name.size() && layer >= 0) {
        return layer * 2 + 1;
    }
    return -1;
}

// ─── Construction ───────────────────────────────────────────────────────

DiskPrefixCache::DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend)
    : config_(cfg), backend_(backend) {}

// ─── Initialization ─────────────────────────────────────────────────────

bool DiskPrefixCache::init() {
    if (disabled()) return true;

    if (!mkdir_p(config_.cache_dir)) {
        std::fprintf(stderr, "[disk-cache] failed to create dir: %s\n",
                     config_.cache_dir.c_str());
        return false;
    }

    // Try to learn layout from existing files (enables first-request disk hits).
    try_learn_from_disk();

    std::fprintf(stderr, "[disk-cache] initialized dir=%s budget=%.1f GB layout=%s\n",
                 config_.cache_dir.c_str(),
                 (double)config_.budget_bytes / (1024.0 * 1024.0 * 1024.0),
                 layout_known_ ? hex(layout_id_.data(), 16).c_str() : "pending");
    return true;
}

// ─── Layout fingerprint ─────────────────────────────────────────────────

void DiskPrefixCache::compute_layout_id(ggml_context * ctx) {
    // Collect tensor metadata sorted by name for deterministic fingerprint.
    struct TInfo { std::string name; uint32_t type; int64_t ne[4]; };
    std::vector<TInfo> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        TInfo ti;
        ti.name = t->name;
        ti.type = (uint32_t)t->type;
        ti.ne[0] = t->ne[0];
        ti.ne[1] = 1;  // normalize sequence-length dimension
        ti.ne[2] = t->ne[2];
        ti.ne[3] = t->ne[3];
        tensors.push_back(std::move(ti));
    }
    std::sort(tensors.begin(), tensors.end(), [](const TInfo & a, const TInfo & b) {
        return a.name < b.name;
    });

    // Build a single buffer and hash it.
    // Prepend identity_salt_ so that config/model differences (model file,
    // max_ctx, chat_template) rotate the layout_id independently of tensor
    // structure. All-zero salt (the default) is back-compatible.
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), identity_salt_.begin(), identity_salt_.end());
    for (const auto & ti : tensors) {
        buf.insert(buf.end(), ti.name.begin(), ti.name.end());
        buf.insert(buf.end(), (uint8_t *)&ti.type, (uint8_t *)&ti.type + 4);
        buf.insert(buf.end(), (uint8_t *)ti.ne, (uint8_t *)ti.ne + 32);
    }

    uint8_t digest[20];
    sha1_hash(buf.data(), buf.size(), digest);
    std::memcpy(layout_id_.data(), digest, 16);
}

void DiskPrefixCache::learn_layout(int slot) {
    if (disabled()) return;
    if (layout_known_ && !layout_from_disk_) return;  // already verified from live model

    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) return;

    std::array<uint8_t, 16> prev_id = layout_id_;
    bool had_disk_layout = layout_from_disk_;

    compute_layout_id(ref.ctx);
    backend_.snapshot_ref_release(slot);

    if (had_disk_layout && std::memcmp(prev_id.data(), layout_id_.data(), 16) != 0) {
        // Model layout differs from what was learned from disk files.
        std::fprintf(stderr, "[disk-cache] layout mismatch: disk=%s model=%s — switching\n",
                     hex(prev_id.data(), 16).c_str(),
                     hex(layout_id_.data(), 16).c_str());
        entries_.clear();
        total_bytes_ = 0;
    }

    layout_known_ = true;
    layout_from_disk_ = false;
    layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
    mkdir_p(layout_dir_);

    std::fprintf(stderr, "[disk-cache] layout learned: %s\n",
                 hex(layout_id_.data(), 16).c_str());

    // Scan for previously saved files matching this layout.
    scan_directory();
}

// ─── Directory scanning ─────────────────────────────────────────────────

void DiskPrefixCache::scan_directory() {
    entries_.clear();
    total_bytes_ = 0;

    if (layout_dir_.empty()) return;

    std::error_code ec;
    for (const auto & de : fs::directory_iterator(layout_dir_, ec)) {
        std::error_code type_ec;
        if (!de.is_regular_file(type_ec) || type_ec) continue;
        const std::string name = de.path().filename().string();
        size_t nlen = name.size();
        if (nlen < 36 || name.compare(nlen - 4, 4, ".dkv") != 0) continue;

        std::string path = de.path().string();
        ParsedDiskFile parsed;
        if (!parse_disk_file(path, parsed)) continue;
        const DiskCacheHeader & hdr = parsed.hdr;

        // Validate magic and layout.
        if (std::memcmp(hdr.magic, "DKVC", 4) != 0) continue;
        if (std::memcmp(hdr.layout_id, layout_id_.data(), 16) != 0) continue;

        DiskEntry entry;
        entry.path = path;
        std::memcpy(entry.token_hash.data(), hdr.token_hash, 16);
        entry.token_count = hdr.token_count;
        entry.cur_pos     = hdr.cur_pos;
        entry.last_used   = hdr.last_used;
        entry.version     = hdr.version;
        entry.reason      = hdr.reason;  // 0 (UNKNOWN) for v1/v2
        entry.parent_hash = parsed.parent_hash;
        entry.parent_tokens = parsed.parent_tokens;
        entry.chain_depth = parsed.chain_depth;

        std::error_code fec;
        auto fsz = fs::file_size(path, fec);
        if (!fec) entry.file_size = (uint64_t)fsz;

        add_total_bytes_saturating(total_bytes_, entry.file_size);
        entries_.push_back(std::move(entry));
    }

    std::fprintf(stderr, "[disk-cache] scanned %zu files, %.1f MB\n",
                 entries_.size(), (double)total_bytes_ / (1024.0 * 1024.0));
}

// ─── Cold start: learn layout from existing files ───────────────────────

void DiskPrefixCache::try_learn_from_disk() {
    // Scan every layout namespace and adopt the one containing the newest
    // valid checkpoint. directory_iterator order is unspecified; returning on
    // its first entry made a restart randomly adopt an obsolete model/layout.
    // When that namespace had no matching prompt, lookup could not validate it
    // and the active cache stayed invisible until the first new snapshot save.
    // At exact-prefill rates that turns a warm 50k-token agent resume into one
    // avoidable cold prefill after every restart.
    std::error_code ec;
    bool found = false;
    uint64_t newest = 0;
    std::array<uint8_t, 16> selected_id{};
    std::string selected_dir;
    // Newest checkpoint per namespace, so the stale-layout reap below reuses
    // this walk instead of parsing every header a second time.
    std::vector<std::pair<std::string, uint64_t>> namespace_newest;
    for (const auto & de : fs::directory_iterator(config_.cache_dir, ec)) {
        const std::string base = de.path().filename().string();
        if (!base.empty() && base[0] == '.') continue;
        std::error_code dec;
        const fs::file_status status = de.symlink_status(dec);
        if (dec || fs::is_symlink(status) || !fs::is_directory(status)) continue;
        const std::string subdir = de.path().string();

        // Check if this subdir has any .dkv files.
        std::error_code sec;
        bool subdir_has_checkpoint = false;
        uint64_t subdir_newest = 0;
        for (const auto & se : fs::directory_iterator(subdir, sec)) {
            std::error_code type_ec;
            if (!se.is_regular_file(type_ec) || type_ec) continue;
            const std::string sname = se.path().filename().string();
            size_t nlen = sname.size();
            if (nlen < 4 || sname.compare(nlen - 4, 4, ".dkv") != 0) continue;

            // Read the header to get the layout_id.
            std::string fpath = se.path().string();
            ParsedDiskFile parsed;
            if (parse_disk_file(fpath, parsed)) {
                const uint64_t when = std::max(
                    parsed.hdr.last_used, parsed.hdr.created_at);
                if (!subdir_has_checkpoint || when > subdir_newest) {
                    subdir_has_checkpoint = true;
                    subdir_newest = when;
                }
                if (!found || when > newest) {
                    found = true;
                    newest = when;
                    std::memcpy(selected_id.data(),
                                parsed.hdr.layout_id, 16);
                    selected_dir = subdir;
                }
            }
        }
        // Only namespaces holding a parseable checkpoint are reap candidates.
        // The sibling continuations-*/tool-memory-* trees store no .dkv and so
        // are never considered.
        if (subdir_has_checkpoint)
            namespace_newest.emplace_back(subdir, subdir_newest);
    }
    if (!found) {
        // Nothing adoptable, but dead namespaces may still be present (every
        // checkpoint corrupt, say). Reaping with no directory to keep is safe.
        reap_stale_layouts_from(namespace_newest, std::string());
        return;
    }
    layout_id_ = selected_id;
    layout_known_ = true;
    layout_from_disk_ = true;  // unverified — confirmed by lookup/save
    layout_dir_ = selected_dir;
    std::fprintf(stderr,
                 "[disk-cache] cold-start selected newest layout %s "
                 "(checkpoint=%" PRIu64 ")\n",
                 hex(layout_id_.data(), 16).c_str(), newest);
    reap_stale_layouts_from(namespace_newest, selected_dir);
    scan_directory();
}

void DiskPrefixCache::reap_stale_layouts_from(
        const std::vector<std::pair<std::string, uint64_t>> & namespaces,
        const std::string & keep_dir) {
    if (config_.stale_layout_days <= 0) return;
    const uint64_t now = (uint64_t) std::time(nullptr);
    const uint64_t cutoff_age =
        (uint64_t) config_.stale_layout_days * 24ull * 60ull * 60ull;

    for (const auto & entry : namespaces) {
        const std::string & dir = entry.first;
        const uint64_t when = entry.second;
        if (!keep_dir.empty() && dir == keep_dir) continue;
        // Clock skew or a future timestamp must never make a namespace look
        // ancient; only a checkpoint genuinely older than the window is reaped.
        if (when > now || now - when < cutoff_age) continue;

        uintmax_t freed = 0;
        std::error_code sz_ec;
        for (const auto & se : fs::directory_iterator(dir, sz_ec)) {
            std::error_code fec;
            if (se.is_regular_file(fec) && !fec) {
                const uintmax_t n = se.file_size(fec);
                if (!fec) freed += n;
            }
        }
        std::error_code rm_ec;
        const uintmax_t removed = fs::remove_all(dir, rm_ec);
        if (rm_ec || removed == 0) {
            std::fprintf(stderr,
                         "[disk-cache] failed to reap stale layout %s: %s\n",
                         dir.c_str(), rm_ec.message().c_str());
            continue;
        }
        std::fprintf(stderr,
                     "[disk-cache] reaped stale layout %s "
                     "(%.1f MB, idle %.1f days)\n",
                     dir.c_str(), (double) freed / (1024.0 * 1024.0),
                     (double) (now - when) / 86400.0);
    }
}

// ─── Lookup ─────────────────────────────────────────────────────────────

bool DiskPrefixCache::lookup(const std::vector<int32_t> & prompt_ids, int slot) {
    if (disabled() || !layout_known_ ||
        !disk_token_count_fits(prompt_ids.size())) return false;

    PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::lock_guard<std::mutex> lock(mu_);
    int idx = find_entry(hash);
    if (idx < 0) return false;

    auto & entry = entries_[idx];
    if (!read_file(entry.path, slot)) {
        // File is corrupt or incompatible — remove it.
        std::fprintf(stderr, "[disk-cache] lookup failed, removing %s\n",
                     entry.path.c_str());
        std::remove(entry.path.c_str());
        total_bytes_ = entry.file_size > total_bytes_
            ? 0 : total_bytes_ - entry.file_size;
        entries_.erase(entries_.begin() + idx);
        return false;
    }

    if (layout_from_disk_) {
        const std::array<uint8_t, 16> disk_id = layout_id_;
        auto ref = backend_.snapshot_ref(slot);
        if (!ref.ctx) {
            backend_.snapshot_free(slot);
            return false;
        }
        compute_layout_id(ref.ctx);
        backend_.snapshot_ref_release(slot);
        if (std::memcmp(disk_id.data(), layout_id_.data(), 16) != 0) {
            std::fprintf(stderr,
                         "[disk-cache] adopted layout mismatch: disk=%s model=%s\n",
                         hex(disk_id.data(), 16).c_str(),
                         hex(layout_id_.data(), 16).c_str());
            backend_.snapshot_free(slot);
            entries_.clear();
            total_bytes_ = 0;
            layout_from_disk_ = false;
            layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
            mkdir_p(layout_dir_);
            scan_directory();
            return false;
        }
        layout_from_disk_ = false;
        layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
    }

    // Update last_used on disk.
    entry.last_used = now_unix();
    entry.hits++;
    // Optionally rewrite header timestamp (non-critical, skip for perf).
    return true;
}

// ─── Save ───────────────────────────────────────────────────────────────

bool DiskPrefixCache::save(int slot, const std::vector<int32_t> & prompt_ids,
                           uint8_t reason) {
    if (disabled() || !disk_token_count_fits(prompt_ids.size())) return false;

    // Learn from a live snapshot on first save. A cold-start layout adopted
    // from an arbitrary existing directory remains unverified until either a
    // matching lookup or this save; never write current-model tensors into
    // that adopted namespace.
    if (!layout_known_ || layout_from_disk_) {
        learn_layout(slot);
        if (!layout_known_ || layout_from_disk_) return false;
    }

    // Check minimum token threshold.
    if ((int)prompt_ids.size() < config_.min_tokens) return false;

    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) return false;
    struct SnapshotRefGuard {
        ModelBackend & backend;
        int slot;
        ~SnapshotRefGuard() { backend.snapshot_ref_release(slot); }
    } ref_guard{backend_, slot};
    if (ref.cur_pos < 0 ||
        (size_t)ref.cur_pos != prompt_ids.size()) {
        std::fprintf(
            stderr,
            "[disk-cache] refusing mismatched checkpoint: snapshot_pos=%d tokens=%zu\n",
            ref.cur_pos, prompt_ids.size());
        return false;
    }

    PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::lock_guard<std::mutex> lock(mu_);

    // Skip if already on disk.
    if (find_entry(hash) >= 0) return true;

    // Choose the deepest stored token prefix as the delta parent.  The chain is
    // bounded so a hot restore never turns into an unbounded sequence of reads.
    const DiskEntry * parent = nullptr;
    // Delta checkpoints currently depend on DeepSeek's per-layer compressor
    // counters in ds4_snap_meta. Qwen exports complete vector-backed state and
    // therefore writes independent base checkpoints until it has an equally
    // explicit append-only counter contract.
    const bool delta_capable =
        ggml_get_tensor(ref.ctx, "ds4_snap_meta") != nullptr;
    for (const auto & candidate : entries_) {
        if (!delta_capable) break;
        const int n = (int)candidate.token_count;
        if (candidate.version != DISK_CACHE_VERSION ||
            n <= 0 || n >= (int)prompt_ids.size() ||
            candidate.cur_pos > (uint32_t)ref.cur_pos ||
            candidate.chain_depth >= DISK_CACHE_MAX_CHAIN_DEPTH) {
            continue;
        }
        if (hash_prefix(prompt_ids.data(), n) != candidate.token_hash) continue;
        if (!parent || candidate.token_count > parent->token_count) {
            parent = &candidate;
        }
    }

    std::string path = make_path(hash);
    std::string tmp_path = path + ".tmp";

    if (!write_file(tmp_path, ref, prompt_ids, parent, reason)) {
        std::remove(tmp_path.c_str());
        return false;
    }

    std::error_code tmp_ec;
    const uint64_t tmp_size = (uint64_t)fs::file_size(tmp_path, tmp_ec);
    if (tmp_ec ||
        (config_.budget_bytes > 0 && tmp_size > config_.budget_bytes)) {
        if (!tmp_ec) {
            std::fprintf(stderr,
                         "[disk-cache] skip save: checkpoint %.1f MB exceeds budget\n",
                         (double)tmp_size / (1024.0 * 1024.0));
        }
        std::remove(tmp_path.c_str());
        return false;
    }

    // Atomic rename.
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return false;
    }
    if (!sync_directory(layout_dir_)) {
        std::fprintf(stderr,
                     "[disk-cache] warning: failed to sync checkpoint directory: %s\n",
                     std::strerror(errno));
    }

    // Update index.
    DiskEntry entry;
    entry.path        = path;
    std::memcpy(entry.token_hash.data(), hash.data(), 16);
    entry.token_count = (uint32_t)prompt_ids.size();
    entry.cur_pos     = (uint32_t)ref.cur_pos;
    entry.last_used   = now_unix();
    entry.created_at  = entry.last_used;
    entry.version     = DISK_CACHE_VERSION;
    entry.reason      = reason;
    if (parent) {
        entry.parent_hash = parent->token_hash;
        entry.parent_tokens = parent->token_count;
        entry.chain_depth = parent->chain_depth + 1;
    }
    std::error_code fec;
    auto fsz = fs::file_size(path, fec);
    if (!fec) entry.file_size = (uint64_t)fsz;

    add_total_bytes_saturating(total_bytes_, entry.file_size);
    entries_.push_back(std::move(entry));

    std::fprintf(stderr,
                 "[disk-cache] saved %s (%u tokens, %d pos, %.1f MB, %s depth=%u)\n",
                 hex(hash.data(), 16).c_str(),
                 (uint32_t)prompt_ids.size(), ref.cur_pos,
                 (double)entries_.back().file_size / (1024.0 * 1024.0),
                 parent ? "delta" : "base", entries_.back().chain_depth);

    enforce_budget(&prompt_ids);
    return true;
}

// ─── Prefix lookup ──────────────────────────────────────────────────────

int DiskPrefixCache::longest_prefix_len(const std::vector<int32_t> & prompt_ids) {
    if (disabled() || !layout_known_ ||
        !disk_token_count_fits(prompt_ids.size())) return 0;
    const int prompt_len = (int)prompt_ids.size();
    int best = 0;
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto & e : entries_) {
        const int n = (int)e.token_count;
        if (n <= best || n > prompt_len || n < config_.min_tokens) continue;
        if (hash_prefix(prompt_ids.data(), n) == e.token_hash) best = n;
    }
    return best;
}

// ─── Budget enforcement ─────────────────────────────────────────────────

void DiskPrefixCache::enforce_budget(const std::vector<int32_t> * incoming_tokens) {
    uint64_t now = now_unix();

    // Reason-aware demotion set: a CONTINUED entry that is a strict token-prefix
    // of the just-saved store is superseded (a wider checkpoint now covers it).
    // Computed once up front; token_hash keys stay valid as entries erase.
    // (ds4_kvstore kv_cache_incoming_supersedes_continued.)
    std::set<PrefixHash> superseded_continued;
    if (incoming_tokens && disk_token_count_fits(incoming_tokens->size())) {
        const int in_n = (int)incoming_tokens->size();
        for (const auto & e : entries_) {
            if (e.reason != DKV_REASON_CONTINUED) continue;
            const int n = (int)e.token_count;
            if (n <= 0 || n >= in_n) continue;
            if (hash_prefix(incoming_tokens->data(), n) == e.token_hash)
                superseded_continued.insert(e.token_hash);
        }
    }

    // DS4-style eviction scoring: (effective_hits + 1) * tokens / file_size with
    // exponential decay on hits (6-hour half-life), then reason weighting.
    auto score = [&](const DiskEntry & e) -> double {
        // Protect recently-saved entries (< 60 seconds old).
        if (e.created_at > 0 && now > 0 && (now - e.created_at) < 60) {
            return 1e18;
        }
        // Decay hits: half-life 6h → decay_rate = ln(2) / (6*3600) ≈ 3.2e-5.
        double age_s = (now > e.last_used) ? (double)(now - e.last_used) : 0.0;
        double effective_hits = (double)e.hits * std::exp(-age_s * 3.2e-5);
        // Floor: a decayed-to-noise hit count reads as never-hit, so stale
        // waypoints become cheap victims (ds4 KV_CACHE_MIN_EFFECTIVE_HITS).
        if (effective_hits < KV_CACHE_MIN_EFFECTIVE_HITS) effective_hits = 0.0;
        double size_factor = (e.file_size > 0) ? (double)e.file_size : 1.0;
        double s = (effective_hits + 1.0) * (double)e.token_count / size_factor;
        // Anchors (cold/evict/shutdown) are expensive to rebuild: 2x harder to
        // evict, so they outlive routine waypoints under budget pressure.
        if (e.reason == DKV_REASON_COLD || e.reason == DKV_REASON_EVICT ||
            e.reason == DKV_REASON_SHUTDOWN) {
            s *= KV_CACHE_ANCHOR_REASON_SCORE_FACTOR;
        }
        // A superseded CONTINUED waypoint is demoted toward eviction, scaled by
        // how hot it still is (hot ones keep more of their score).
        if (superseded_continued.count(e.token_hash)) {
            double h = effective_hits > 0.0
                           ? effective_hits / (effective_hits + 1.0)
                           : 0.0;
            s *= KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR +
                 KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR * h;
        }
        return s;
    };

    while (total_bytes_ > config_.budget_bytes && !entries_.empty()) {
        // Delta parents are live data, not independently evictable cache
        // entries.  Evict leaves first; once their children are gone a former
        // parent naturally becomes eligible on the next pass.
        auto is_parent = [&](const DiskEntry & candidate) {
            for (const auto & e : entries_) {
                if (!hash_is_zero(e.parent_hash) &&
                    e.parent_hash == candidate.token_hash) {
                    return true;
                }
            }
            return false;
        };
        auto it = entries_.end();
        double worst = std::numeric_limits<double>::infinity();
        for (auto cur = entries_.begin(); cur != entries_.end(); ++cur) {
            if (is_parent(*cur)) continue;
            const double s = score(*cur);
            if (s < worst) {
                worst = s;
                it = cur;
            }
        }

        // Recent entries win over stale entries through their high score, but
        // the configured budget remains a hard ceiling. If every leaf is
        // recent, evict the least valuable recent leaf instead of remaining
        // over budget indefinitely after a burst of saves.
        if (it == entries_.end()) break;

        std::fprintf(stderr, "[disk-cache] evicting %s (%.1f MB, hits=%u, score=%.3f)\n",
                     hex(it->token_hash.data(), 16).c_str(),
                     (double)it->file_size / (1024.0 * 1024.0),
                     it->hits, score(*it));

        std::remove(it->path.c_str());
        total_bytes_ = it->file_size > total_bytes_
            ? 0 : total_bytes_ - it->file_size;
        entries_.erase(it);
    }
}

// ─── Touch ──────────────────────────────────────────────────────────────

void DiskPrefixCache::touch(const PrefixHash & hash) {
    std::lock_guard<std::mutex> lock(mu_);
    int idx = find_entry(hash);
    if (idx >= 0) {
        entries_[idx].last_used = now_unix();
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

std::string DiskPrefixCache::make_path(const PrefixHash & hash) const {
    return layout_dir_ + "/" + hex(hash.data(), 16) + ".dkv";
}

int DiskPrefixCache::find_entry(const PrefixHash & hash) const {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[i].token_hash == hash) return i;
    }
    return -1;
}

// ─── File I/O: Write ────────────────────────────────────────────────────

bool DiskPrefixCache::write_file(const std::string & path,
                                 const ModelBackend::SnapshotRef & ref,
                                 const std::vector<int32_t> & prompt_ids,
                                 const DiskEntry * parent,
                                 uint8_t reason) {
    ParsedDiskFile parent_file;
    std::vector<int32_t> parent_meta;
    std::vector<int32_t> current_meta;
    if (parent) {
        if (!parse_disk_file(parent->path, parent_file) ||
            std::memcmp(parent_file.hdr.layout_id, layout_id_.data(), 16) != 0 ||
            parent_file.hdr.token_count != parent->token_count ||
            parent_file.chain_depth != parent->chain_depth ||
            !read_file_snapshot_meta(parent->path, parent_file, parent_meta) ||
            !read_context_snapshot_meta(ref.ctx, current_meta) ||
            parent_meta.size() != current_meta.size()) {
            return false;
        }
    }

    std::vector<DiskTensorPatch> table;
    uint64_t payload_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t;
         t = ggml_get_next_tensor(ref.ctx, t)) {
        DiskTensorPatch patch;
        patch.tensor.name = t->name;
        patch.tensor.type = (uint32_t)t->type;
        for (int d = 0; d < 4; ++d) patch.tensor.ne[d] = t->ne[d];
        patch.tensor.nbytes = ggml_nbytes(t);
        patch.patch_offset = 0;
        patch.patch_bytes = patch.tensor.nbytes;

        if (parent && snapshot_tensor_can_grow(patch.tensor.name)) {
            auto it = std::find_if(
                parent_file.tensors.begin(), parent_file.tensors.end(),
                [&](const DiskTensorPatch & p) {
                    return p.tensor.name == patch.tensor.name;
                });
            if (it != parent_file.tensors.end() &&
                tensor_shape_prefix(it->tensor, patch.tensor)) {
                const int counter = compressed_counter_index(
                    patch.tensor.name);
                if (counter >= 0 &&
                    (size_t)counter < parent_meta.size() &&
                    parent_meta[(size_t)counter] >= 0 &&
                    parent_meta[(size_t)counter] <=
                        current_meta[(size_t)counter] &&
                    parent_meta[(size_t)counter] <= it->tensor.ne[1] &&
                    patch.tensor.ne[1] > 0 &&
                    patch.tensor.nbytes %
                        (uint64_t)patch.tensor.ne[1] == 0) {
                    const uint64_t row_bytes =
                        patch.tensor.nbytes /
                        (uint64_t)patch.tensor.ne[1];
                    patch.patch_offset =
                        (uint64_t)parent_meta[(size_t)counter] *
                        row_bytes;
                    patch.patch_bytes =
                        patch.tensor.nbytes - patch.patch_offset;
                }
            }
        }
        if (payload_bytes > UINT64_MAX - patch.patch_bytes) return false;
        payload_bytes += patch.patch_bytes;
        table.push_back(std::move(patch));
    }
    if (table.empty() || table.size() > UINT32_MAX) return false;

    FILE * f = open_write_no_follow(path);
    if (!f) return false;

    // Write header.
    DiskCacheHeader hdr{};
    std::memcpy(hdr.magic, "DKVC", 4);
    hdr.version = DISK_CACHE_VERSION;
    std::memcpy(hdr.layout_id, layout_id_.data(), 16);
    hdr.cur_pos       = (uint32_t)ref.cur_pos;
    hdr.n_tensors     = (uint32_t)table.size();
    hdr.token_count   = (uint32_t)prompt_ids.size();
    PrefixHash ph = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    std::memcpy(hdr.token_hash, ph.data(), 16);
    hdr.payload_bytes = payload_bytes;
    hdr.created_at    = now_unix();
    hdr.last_used     = hdr.created_at;
    hdr.last_tok      = ref.last_tok;
    hdr.reason        = reason;

    if (!write_header(f, hdr)) { std::fclose(f); return false; }

    PrefixHash parent_hash{};
    uint32_t parent_tokens = 0;
    uint32_t chain_depth = 0;
    if (parent) {
        parent_hash = parent->token_hash;
        parent_tokens = parent->token_count;
        chain_depth = parent->chain_depth + 1;
    }
    if (std::fwrite(parent_hash.data(), 1, parent_hash.size(), f) !=
            parent_hash.size()) {
        std::fclose(f);
        return false;
    }
    write_u32(f, parent_tokens);
    write_u32(f, chain_depth);

    // Write tensor table.
    for (const auto & patch : table) {
        uint16_t name_len = (uint16_t)patch.tensor.name.size();
        write_u16(f, name_len);
        std::fwrite(patch.tensor.name.data(), 1, name_len, f);
        write_u32(f, patch.tensor.type);
        for (int d = 0; d < 4; ++d) write_i64(f, patch.tensor.ne[d]);
        write_u64(f, patch.tensor.nbytes);
        write_u64(f, patch.patch_offset);
        write_u64(f, patch.patch_bytes);
    }
    if (std::ferror(f)) { std::fclose(f); return false; }

    // Write only each tensor's changed span.  Compressed-KV rows are append-only
    // within a conversation; rolling rings, compressor state, HC and logits are
    // deliberately complete patches because they mutate in place.
    std::vector<uint8_t> buf(4 * 1024 * 1024);  // 4 MB transfer buffer
    size_t tensor_index = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t;
         t = ggml_get_next_tensor(ref.ctx, t), ++tensor_index) {
        const auto & patch = table[tensor_index];
        uint64_t offset = 0;
        while (offset < patch.patch_bytes) {
            size_t chunk = (size_t)std::min<uint64_t>(
                buf.size(), patch.patch_bytes - offset);
            ggml_backend_tensor_get(
                t, buf.data(), (size_t)(patch.patch_offset + offset), chunk);
            if (std::fwrite(buf.data(), 1, chunk, f) != chunk) {
                std::fclose(f);
                return false;
            }
            offset += chunk;
        }
    }

    // Commit the complete temporary file before the caller atomically renames
    // it into the content-addressed namespace.
    bool ok = std::fflush(f) == 0;
    if (ok) ok = ::fsync(::fileno(f)) == 0;
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

// ─── File I/O: Read ─────────────────────────────────────────────────────

bool DiskPrefixCache::read_file(const std::string & path, int slot) {
    ParsedDiskFile top;
    if (!parse_disk_file(path, top) ||
        std::memcmp(top.hdr.layout_id, layout_id_.data(), 16) != 0) {
        return false;
    }

    // Allocate ggml context + buffer.
    ggml_init_params ip{};
    ip.mem_size =
        ggml_tensor_overhead() * (size_t)(top.hdr.n_tensors + 4) + 4096;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    // Create tensors.
    for (const auto & patch : top.tensors) {
        const auto & ent = patch.tensor;
        ggml_tensor * t =
            ggml_new_tensor(ctx, (ggml_type)ent.type, 4, ent.ne);
        if (!t) {
            ggml_free(ctx);
            return false;
        }
        if (ggml_nbytes(t) != ent.nbytes) {
            ggml_free(ctx);
            return false;
        }
        ggml_set_name(t, ent.name.c_str());
    }

    // Allocate buffer on CPU backend.
    // #6: Reuse the shared process-wide CPU backend instead of creating (and
    // leaking) a new one per load. It is owned by disk_cache_cpu_backend()'s
    // static holder, so it must NOT be freed here or in any error path below.
    ggml_backend_t cpu = disk_cache_cpu_backend();
    if (!cpu) { ggml_free(ctx); return false; }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_clear(buf, 0);

    // Replay base -> suffix patches directly into the final CPU tensors.  This
    // avoids materializing a second full snapshot in temporary host vectors.
    std::vector<uint8_t> read_buf(4 * 1024 * 1024);
    std::function<bool(const std::string &, int)> apply =
        [&](const std::string & current_path, int recursion) -> bool {
        if (recursion > DISK_CACHE_MAX_CHAIN_DEPTH) return false;
        ParsedDiskFile file;
        if (!parse_disk_file(current_path, file) ||
            std::memcmp(file.hdr.layout_id, layout_id_.data(), 16) != 0) {
            return false;
        }

        if (!hash_is_zero(file.parent_hash)) {
            int parent_idx = find_entry(file.parent_hash);
            if (parent_idx < 0) return false;
            ParsedDiskFile parent_file;
            if (!parse_disk_file(entries_[(size_t)parent_idx].path,
                                 parent_file) ||
                parent_file.hdr.token_count != file.parent_tokens ||
                parent_file.chain_depth + 1 != file.chain_depth ||
                std::memcmp(parent_file.hdr.token_hash,
                            file.parent_hash.data(),
                            file.parent_hash.size()) != 0 ||
                !apply(entries_[(size_t)parent_idx].path, recursion + 1)) {
                return false;
            }
        }

        FILE * f = open_read_no_follow(current_path);
        if (!f) return false;
        if (std::fseek(f, file.payload_offset, SEEK_SET) != 0) {
            std::fclose(f);
            return false;
        }
        for (const auto & patch : file.tensors) {
            ggml_tensor * target =
                ggml_get_tensor(ctx, patch.tensor.name.c_str());
            bool shape_ok = target != nullptr;
            if (shape_ok) {
                for (int d = 0; d < 4; ++d) {
                    const bool growing_dim =
                        d == 1 &&
                        snapshot_tensor_can_grow(patch.tensor.name);
                    if ((growing_dim && patch.tensor.ne[d] > target->ne[d]) ||
                        (!growing_dim && patch.tensor.ne[d] != target->ne[d])) {
                        shape_ok = false;
                        break;
                    }
                }
            }
            if (!shape_ok ||
                target->type != (ggml_type)patch.tensor.type ||
                patch.tensor.nbytes > ggml_nbytes(target) ||
                patch.patch_offset + patch.patch_bytes >
                    ggml_nbytes(target)) {
                std::fclose(f);
                return false;
            }
            uint64_t offset = 0;
            while (offset < patch.patch_bytes) {
                size_t chunk = (size_t)std::min<uint64_t>(
                    read_buf.size(), patch.patch_bytes - offset);
                if (std::fread(read_buf.data(), 1, chunk, f) != chunk) {
                    std::fclose(f);
                    return false;
                }
                ggml_backend_tensor_set(
                    target, read_buf.data(),
                    (size_t)(patch.patch_offset + offset), chunk);
                offset += chunk;
            }
        }
        std::fclose(f);
        return true;
    };

    if (!apply(path, 0)) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // Hand off to backend.
    if (!backend_.snapshot_adopt(slot, ctx, buf, (int)top.hdr.cur_pos,
                                 top.hdr.last_tok)) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);  // #6: shared cpu backend is not freed per-load
        return false;
    }

    // #6: The CPU backend here is the shared process-wide instance returned by
    // disk_cache_cpu_backend(); it outlives this buffer and every other adopted
    // snapshot buffer, so it is intentionally NOT freed here. Only buf + ctx are
    // handed to the backend (freed later by snapshot_free); the single shared
    // cpu backend is released once, at process exit, by its static holder.

    return true;
}

// ─── Header I/O ─────────────────────────────────────────────────────────

bool DiskPrefixCache::write_header(FILE * f, const DiskCacheHeader & hdr) {
    std::fwrite(hdr.magic, 1, 4, f);
    write_u32(f, hdr.version);
    std::fwrite(hdr.layout_id, 1, 16, f);
    write_u32(f, hdr.cur_pos);
    write_u32(f, hdr.n_tensors);
    write_u32(f, hdr.token_count);
    std::fwrite(hdr.token_hash, 1, 16, f);
    write_u64(f, hdr.payload_bytes);
    write_u64(f, hdr.created_at);
    write_u64(f, hdr.last_used);
    write_u32(f, (uint32_t)hdr.last_tok);  // stored as u32, cast back on read
    // Base header total: 4+4+16+4+4+4+16+8+8+8+4 = 80 bytes.
    // v3+ appends a 4-byte reason block (reason/ext_flags/reserved).
    if (hdr.version >= DISK_CACHE_VERSION) {
        write_u8(f, hdr.reason);
        write_u8(f, hdr.ext_flags);
        write_u16(f, hdr.reserved);
    }
    return !std::ferror(f);
}

bool DiskPrefixCache::read_header(FILE * f, DiskCacheHeader & hdr) {
    if (std::fread(hdr.magic, 1, 4, f) != 4) return false;
    if (!read_u32(f, hdr.version)) return false;
    if (std::fread(hdr.layout_id, 1, 16, f) != 16) return false;
    if (!read_u32(f, hdr.cur_pos)) return false;
    if (!read_u32(f, hdr.n_tensors)) return false;
    if (!read_u32(f, hdr.token_count)) return false;
    if (std::fread(hdr.token_hash, 1, 16, f) != 16) return false;
    if (!read_u64(f, hdr.payload_bytes)) return false;
    if (!read_u64(f, hdr.created_at)) return false;
    if (!read_u64(f, hdr.last_used)) return false;
    uint32_t last_tok_raw;
    if (!read_u32(f, last_tok_raw)) return false;
    hdr.last_tok = (int32_t)last_tok_raw;
    if (hdr.version >= DISK_CACHE_VERSION) {
        if (!read_u8(f, hdr.reason) || !read_u8(f, hdr.ext_flags) ||
            !read_u16(f, hdr.reserved)) {
            return false;
        }
    }
    return true;
}

}  // namespace dflash::common
