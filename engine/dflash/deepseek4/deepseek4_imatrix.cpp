#include "deepseek4_imatrix.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace ds4 {

ImatrixCollector * ImatrixCollector::instance() {
    static ImatrixCollector * collector = [] () -> ImatrixCollector * {
        const char * out = std::getenv("DFLASH_IMATRIX_OUT");
        if (!out || !*out) return nullptr;
        return new ImatrixCollector();
    }();
    return collector;
}

ImatrixCollector::Entry * ImatrixCollector::entry_mut(const std::string & name, int n_in) {
    for (auto & kv : entries_) {
        if (kv.first == name) return &kv.second;
    }
    Entry e;
    e.n_in = n_in;
    e.values.assign((size_t) n_in * (size_t) n_expert_, 0.0);
    e.counts.assign((size_t) n_expert_, 0);
    entries_.emplace_back(name, std::move(e));
    return &entries_.back().second;
}

const ImatrixCollector::Entry * ImatrixCollector::entry(const std::string & name) const {
    for (const auto & kv : entries_) {
        if (kv.first == name) return &kv.second;
    }
    return nullptr;
}

void ImatrixCollector::accumulate(const std::string & name, const float * input,
                                  const int32_t * ids, int n_in, int n_used,
                                  int n_tokens, bool per_slot) {
    Entry * e = entry_mut(name, n_in);
    if (e->n_in != n_in) return;   // shape drift: refuse rather than corrupt
    for (int t = 0; t < n_tokens; ++t) {
        for (int s = 0; s < n_used; ++s) {
            const int ex = ids[(size_t) t * n_used + s];
            if (ex < 0 || ex >= n_expert_) continue;
            // gate/up see the token's own row once per expert it routed to;
            // down sees a different row per (slot, token), because the row is
            // the SwiGLU output of that particular expert.
            const float * x = per_slot
                ? input + (((size_t) t * n_used) + s) * n_in
                : input + (size_t) t * n_in;
            double * v = e->values.data() + (size_t) ex * n_in;
            for (int j = 0; j < n_in; ++j) {
                v[j] += (double) x[j] * (double) x[j];
            }
            e->counts[ex] += 1;
        }
    }
    e->ncall += 1;
}

void ImatrixCollector::abandon() {
    sites_.clear();
}

void ImatrixCollector::end_chunk() {
    chunks_ += 1;
}

static bool write_all(FILE * f, const void * p, size_t n) {
    return std::fwrite(p, 1, n, f) == n;
}

bool ImatrixCollector::write(const std::string & path, std::string & err) const {
    const std::string tmp = path + ".part";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "open " + tmp + ": " + std::strerror(errno); return false; }

    bool ok = true;
    const int32_t n_entries = (int32_t) entries_.size();
    ok = ok && write_all(f, &n_entries, sizeof(n_entries));
    for (const auto & kv : entries_) {
        const Entry & e = kv.second;
        const int32_t len = (int32_t) kv.first.size();
        ok = ok && write_all(f, &len, sizeof(len));
        ok = ok && write_all(f, kv.first.data(), kv.first.size());
        ok = ok && write_all(f, &e.ncall, sizeof(int32_t));
        const int32_t nval = (int32_t) e.values.size();
        ok = ok && write_all(f, &nval, sizeof(nval));
        // The .dat stores values already divided by counts. An expert nothing
        // routed to keeps 0 rather than a 1.0 sentinel: a fabricated row is
        // indistinguishable from a measured one downstream, and llama-quantize
        // does not warn. Audit the counts before shipping a matrix.
        std::vector<float> row(e.values.size());
        for (int ex = 0; ex < n_expert_; ++ex) {
            const double c = e.counts[ex] > 0 ? (double) e.counts[ex] : 0.0;
            for (int j = 0; j < e.n_in; ++j) {
                const size_t i = (size_t) ex * e.n_in + j;
                row[i] = c > 0.0 ? (float) (e.values[i] / c) : 0.0f;
            }
        }
        ok = ok && write_all(f, row.data(), sizeof(float) * row.size());
    }
    const int32_t last_chunk = chunks_;
    ok = ok && write_all(f, &last_chunk, sizeof(last_chunk));
    const int32_t dlen = (int32_t) dataset_.size();
    ok = ok && write_all(f, &dlen, sizeof(dlen));
    ok = ok && write_all(f, dataset_.data(), dataset_.size());

    if (ok) ok = std::fflush(f) == 0;
    if (ok) ok = fsync(fileno(f)) == 0;
    std::fclose(f);
    if (!ok) {
        err = "write " + tmp + ": " + std::strerror(errno);
        std::remove(tmp.c_str());
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "rename " + tmp + ": " + std::strerror(errno);
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

}  // namespace ds4
