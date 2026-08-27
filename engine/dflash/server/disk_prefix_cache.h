// Disk-backed prefix cache — persists KV snapshots to disk.
//
// Serializes backend snapshot tensors to files, enabling cache survival across
// restarts and overflow beyond Ember's in-memory snapshot slots.
//
// File format v1: 80-byte header + tensor table + raw tensor data.
// File format v2: the same header, a parent-prefix descriptor, then a tensor
// table whose entries describe byte patches.  Growing compressed-KV tensors
// store only the suffix beyond the parent checkpoint; fixed rolling state is
// written in full.  A bounded parent chain is materialized transactionally on
// restore.  Existing v1 files remain readable.
// File format v3: adds a 4-byte reason block (reason/ext_flags/reserved) after
// the 80-byte base header and before the v2 parent descriptor, persisting the
// save reason (cold anchor / continued waypoint / routine) so reason-aware
// eviction survives a restart (port of ds4_kvstore reasons). v1/v2 files remain
// readable and are treated as reason=UNKNOWN.
// Files are keyed by SHA-1 of prompt token IDs.
// A layout fingerprint (SHA-1 of tensor names/types/shapes) prevents loading
// snapshots from incompatible models.
//
// Directory structure:
//   <cache_dir>/<layout_fingerprint_hex>/<token_hash_hex>.dkv

#pragma once

#include "common/model_backend.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace dflash::common {

using PrefixHash = std::array<uint8_t, 16>;

// ─── Configuration ──────────────────────────────────────────────────────

struct DiskCacheConfig {
    std::string cache_dir;          // base directory (empty = disabled)
    // 128 GB default. The old 4 GB was far too small for what this stores: a
    // 45-82k-token context serialises to a 100-450 MB snapshot, so 4 GB held
    // only a handful and evicted entries that each turn a multi-minute cold
    // prefill into a restore (measured: ~15 min of silent prefill for 63k, vs
    // 98-100% cache hits once snapshots survive). Entries are scored by
    // (hits+1)*tokens/size with a 6h half-life, so a large budget retains
    // useful snapshots rather than hoarding — stale ones still age out.
    // Override per-deployment with --kv-cache-mb.
    size_t      budget_bytes = (size_t)128 * 1024 * 1024 * 1024;  // 128 GB default
    int         min_tokens   = 512; // only persist snapshots >= this many tokens
    // Stale-layout reaping, in days (0 disables).
    //
    // The cache is partitioned by layout fingerprint, and everything below --
    // scan_directory(), total_bytes_, enforce_budget() -- is scoped to the ONE
    // active namespace. So when the fingerprint changes (new model, requant,
    // an engine change that moves tensor layout) the entire previous namespace
    // is stranded: never scanned, never counted against the budget, never
    // evicted, never reused. Only a human ever removed it. Measured on
    // 2026-08-05: six dead namespaces, 45.8 GB, accumulated in five days --
    // and because they are invisible to total_bytes_, `du` disagreed with the
    // tracked size badly enough to mislead a capacity diagnosis.
    //
    // Reaped at init on recency, not on "is it the active one": a fingerprint
    // change is often a rollback-able experiment, so the previous namespace
    // must survive long enough to be switched back to. Only namespaces whose
    // newest checkpoint is older than this are removed.
    int         stale_layout_days = 7;
};

// ─── File header (80 bytes, little-endian) ──────────────────────────────

struct DiskCacheHeader {
    char     magic[4];          // "DKVC"
    uint32_t version;           // 1 (legacy full snapshot) or 2 (delta chain)
    uint8_t  layout_id[16];    // SHA-1 truncated: tensor structure fingerprint
    uint32_t cur_pos;
    uint32_t n_tensors;
    uint32_t token_count;      // number of prompt tokens
    uint8_t  token_hash[16];   // SHA-1 of prompt token IDs (same as PrefixHash)
    uint64_t payload_bytes;    // total tensor data bytes
    uint64_t created_at;       // unix seconds
    uint64_t last_used;        // unix seconds (updated on hit)
    int32_t  last_tok;         // last prefill token (needed for decode seeding)
    // v3+ only on disk (4-byte block after the 80-byte base header). v1/v2 files
    // have no such block and decode as reason=UNKNOWN.
    uint8_t  reason = 0;       // DiskCacheReason (governs eviction weighting)
    uint8_t  ext_flags = 0;    // reserved for future per-entry flags
    uint16_t reserved = 0;     // pad the reason block to 4 bytes
};
// NOTE: The header is serialized field-by-field, not as a raw struct, to avoid
// alignment/packing issues. The base header is exactly 80 bytes; v3 appends a
// 4-byte reason block (84 bytes total before the parent descriptor).

static constexpr size_t DISK_CACHE_HEADER_SIZE = 80;
static constexpr uint32_t DISK_CACHE_VERSION_V1 = 1;
static constexpr uint32_t DISK_CACHE_VERSION_V2 = 2;  // delta chain, no reason byte
static constexpr uint32_t DISK_CACHE_VERSION = 3;     // + persisted save reason
static constexpr int DISK_CACHE_MAX_CHAIN_DEPTH = 8;

// Per-entry save reason, persisted in v3+ headers and mirrored from ds4_kvstore.
// Governs eviction weighting only (never correctness). Anchors (COLD/EVICT/
// SHUTDOWN) are expensive-to-rebuild shared prefixes and are protected;
// CONTINUED are mid-generation waypoints a later superset store can demote.
enum DiskCacheReason : uint8_t {
    DKV_REASON_UNKNOWN   = 0,  // routine waypoint (turn boundary, tool-call)
    DKV_REASON_COLD      = 1,  // cold system-prefix anchor (eviction-protected)
    DKV_REASON_CONTINUED = 2,  // mid-generation periodic checkpoint
    DKV_REASON_EVICT     = 3,  // (reserved) pre-eviction anchor
    DKV_REASON_SHUTDOWN  = 4,  // (reserved) shutdown anchor
};

// Eviction-scoring constants (ds4_kvstore.c parity: values verified against
// ds4 @ 54b36ed). Anchors score 2x (harder to evict); a decayed hit count below
// the floor is treated as zero; a CONTINUED entry that the incoming store is a
// token-superset of is demoted by MIN + HIT*h (h = hot-ness in [0,1)).
static constexpr double KV_CACHE_ANCHOR_REASON_SCORE_FACTOR   = 2.0;
static constexpr double KV_CACHE_MIN_EFFECTIVE_HITS           = 0.01;
static constexpr double KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR  = 0.05;
static constexpr double KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR  = 0.45;

// ─── Tensor table entry (on-disk) ──────────────────────────────────────

struct DiskTensorEntry {
    std::string name;
    uint32_t    type;          // ggml_type enum
    int64_t     ne[4];
    size_t      nbytes;
};

// ─── DiskPrefixCache ────────────────────────────────────────────────────

class DiskPrefixCache {
public:
    DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend);
    ~DiskPrefixCache() = default;

    DiskPrefixCache(const DiskPrefixCache &) = delete;
    DiskPrefixCache & operator=(const DiskPrefixCache &) = delete;

    bool disabled() const { return config_.cache_dir.empty(); }

    // Initialize: create directory, scan existing files, learn layout from
    // first available snapshot. Returns false on fatal error.
    bool init();

    // Look up a prompt on disk. On hit, loads the snapshot into `out_slot`
    // using backend.snapshot_adopt(). Returns true if loaded successfully.
    bool lookup(const std::vector<int32_t> & prompt_ids, int slot);

    // Save the snapshot in `slot` to disk, keyed by prompt_ids. `reason` is
    // persisted (v3+) and steers eviction weighting; default UNKNOWN = routine.
    // Returns true on success.
    bool save(int slot, const std::vector<int32_t> & prompt_ids,
              uint8_t reason = DKV_REASON_UNKNOWN);

    // Evict files until total disk usage is within budget. `incoming_tokens`,
    // when given, is the just-saved store; a CONTINUED entry that is a strict
    // token-prefix of it is demoted (ds4 incoming-supersedes-continued).
    void enforce_budget(const std::vector<int32_t> * incoming_tokens = nullptr);

    // Update last_used timestamp for a file (on cache hit).
    void touch(const PrefixHash & hash);

    // Get total bytes used on disk.
    size_t total_bytes() const { return total_bytes_; }

    // Longest stored checkpoint that is a prefix of `prompt_ids`, or 0.
    // Port of ds4_kvstore_find_text_prefix: for each entry, re-hash the
    // prompt's first token_count tokens and compare. Callers probing a fixed
    // list of candidate lengths can only find a checkpoint whose length they
    // already guessed, which makes cold/continued entries effectively
    // unreachable — their length depends on where generation stopped.
    int longest_prefix_len(const std::vector<int32_t> & prompt_ids);

    // Learn the layout fingerprint from a live snapshot (call once after
    // first snapshot_save, before any disk operations).
    void learn_layout(int slot);

    // Set a config/model identity salt prepended to the layout hash buffer
    // so that layout_id encodes model identity in addition to tensor structure.
    // Call this BEFORE init(). All-zeroes (the default) → back-compat behavior.
    void set_identity_salt(const std::array<uint8_t, 16> & salt) {
        identity_salt_ = salt;
    }

private:
    DiskCacheConfig config_;
    ModelBackend &  backend_;

    // Config/model identity salt (set via set_identity_salt before init()).
    // All-zeroes by default → backward-compatible behavior.
    std::array<uint8_t, 16> identity_salt_{};

    // Layout fingerprint (learned from first snapshot).
    std::array<uint8_t, 16> layout_id_{};
    bool layout_known_ = false;
    bool layout_from_disk_ = false;  // true if learned from file, unverified
    std::string layout_dir_;  // <cache_dir>/<fingerprint_hex>/

    // In-memory index of on-disk files.
    struct DiskEntry {
        std::string path;
        PrefixHash  token_hash;
        uint32_t    token_count = 0;
        uint32_t    cur_pos     = 0;
        uint64_t    file_size   = 0;
        uint64_t    last_used   = 0;
        uint32_t    hits        = 0;
        uint64_t    created_at  = 0;  // for eviction protection
        uint32_t    version     = DISK_CACHE_VERSION_V1;
        PrefixHash  parent_hash{};
        uint32_t    parent_tokens = 0;
        uint32_t    chain_depth = 0;
        uint8_t     reason      = DKV_REASON_UNKNOWN;  // 0 for v1/v2
    };
    std::vector<DiskEntry> entries_;
    size_t total_bytes_ = 0;
    std::mutex mu_;

    // Helpers.
    void compute_layout_id(ggml_context * ctx);
    void scan_directory();
    void try_learn_from_disk();
    // Remove layout namespaces older than config_.stale_layout_days. Never
    // touches `keep_dir` (the namespace just adopted), never touches a
    // directory holding no parseable checkpoint -- which is what excludes the
    // sibling continuations-*/tool-memory-* trees, since those store no .dkv.
    void reap_stale_layouts_from(
        const std::vector<std::pair<std::string, uint64_t>> & namespaces,
        const std::string & keep_dir);
    std::string make_path(const PrefixHash & hash) const;
    int find_entry(const PrefixHash & hash) const;

    bool write_file(const std::string & path,
                    const ModelBackend::SnapshotRef & ref,
                    const std::vector<int32_t> & prompt_ids,
                    const DiskEntry * parent,
                    uint8_t reason);
    bool read_file(const std::string & path, int slot);

    // Header I/O.
    static bool write_header(FILE * f, const DiskCacheHeader & hdr);
    static bool read_header(FILE * f, DiskCacheHeader & hdr);
};

}  // namespace dflash::common
