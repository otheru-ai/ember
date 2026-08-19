#include "dspark_cpu_ops.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>

namespace ember::xdna2 {
namespace {

constexpr float kHcEpsilon = 1.0e-6f;

size_t u(int value) { return static_cast<size_t>(value); }

class DsparkCpuPool {
public:
    DsparkCpuPool() {
        unsigned count = std::thread::hardware_concurrency();
        if (count == 0) count = 4;
        count = std::min(count, 16u);
        if (const char * raw = std::getenv("EMBER_DSPARK_CPU_THREADS")) {
            char * end = nullptr;
            const long parsed = std::strtol(raw, &end, 10);
            if (end != raw && *end == '\0' && parsed >= 1 && parsed <= 64)
                count = static_cast<unsigned>(parsed);
        }
        for (unsigned index = 1; index < count; ++index)
            workers_.emplace_back([this] { worker(); });
    }

    ~DsparkCpuPool() {
        {
            std::lock_guard<std::mutex> lock(state_lock_);
            stop_ = true;
            ++generation_;
        }
        ready_.notify_all();
        for (std::thread & worker_thread : workers_) worker_thread.join();
    }

    void run(int count, const std::function<void(int)> & function) {
        if (count <= 0) return;
        if (workers_.empty() || count == 1) {
            for (int index = 0; index < count; ++index) function(index);
            return;
        }
        std::lock_guard<std::mutex> client(client_lock_);
        {
            std::lock_guard<std::mutex> lock(state_lock_);
            function_ = function;
            count_ = count;
            next_.store(0, std::memory_order_relaxed);
            pending_ = static_cast<int>(workers_.size());
            ++generation_;
        }
        ready_.notify_all();
        execute(function);
        std::unique_lock<std::mutex> lock(state_lock_);
        complete_.wait(lock, [this] { return pending_ == 0; });
        function_ = {};
    }

private:
    void execute(const std::function<void(int)> & function) {
        for (;;) {
            const int index = next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= count_) return;
            function(index);
        }
    }

    void worker() {
        uint64_t observed = 0;
        for (;;) {
            std::function<void(int)> function;
            {
                std::unique_lock<std::mutex> lock(state_lock_);
                ready_.wait(lock, [this, observed] {
                    return stop_ || generation_ != observed;
                });
                if (stop_) return;
                observed = generation_;
                function = function_;
            }
            execute(function);
            {
                std::lock_guard<std::mutex> lock(state_lock_);
                if (--pending_ == 0) complete_.notify_one();
            }
        }
    }

    std::mutex client_lock_;
    std::mutex state_lock_;
    std::condition_variable ready_;
    std::condition_variable complete_;
    std::vector<std::thread> workers_;
    std::function<void(int)> function_;
    std::atomic<int> next_{0};
    int count_ = 0;
    int pending_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

DsparkCpuPool & cpu_pool() {
    static DsparkCpuPool pool;
    return pool;
}

bool fail(std::string * error, const char * message) {
    if (error) *error = message;
    return false;
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

bool finite_array(const float * values, size_t count) {
    if (!values) return false;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}

void rms_norm_row(const float * input, const float * weight, int width,
                  float epsilon, float * output) {
    double squares = 0.0;
    for (int lane = 0; lane < width; ++lane)
        squares += static_cast<double>(input[lane]) * input[lane];
    const float inverse = 1.0f / std::sqrt(
        static_cast<float>(squares / width) + epsilon);
    for (int lane = 0; lane < width; ++lane)
        output[lane] = input[lane] * inverse * (weight ? weight[lane] : 1.0f);
}

void rope_row(float * row, int head_dim, int rope_dims, int32_t position,
              float rope_base) {
    const int offset = head_dim - rope_dims;
    const double theta_scale = std::pow(
        static_cast<double>(rope_base), -2.0 / static_cast<double>(rope_dims));
    constexpr double tau = 6.2831853071795864769;
    for (int lane = 0; lane < rope_dims; lane += 2) {
        double angle = static_cast<double>(position) *
                       std::pow(theta_scale, lane / 2);
        angle -= tau * std::floor(angle / tau);
        const float cosine = std::cos(static_cast<float>(angle));
        const float sine = std::sin(static_cast<float>(angle));
        const float left = row[offset + lane];
        const float right = row[offset + lane + 1];
        row[offset + lane] = left * cosine - right * sine;
        row[offset + lane + 1] = left * sine + right * cosine;
    }
}

}  // namespace

void dspark_parallel_for(int count,
                         const std::function<void(int)> & function) {
    cpu_pool().run(count, function);
}

bool dspark_weighted_rms_norm(const float * input, const float * weight,
                              int rows, int width, float epsilon,
                              std::vector<float> & output,
                              std::string * error) {
    output.clear();
    if (error) error->clear();
    if (!input || !weight || rows <= 0 || width <= 0 || epsilon < 0.0f ||
        !finite_array(input, u(rows) * u(width)) ||
        !finite_array(weight, u(width)))
        return fail(error, "invalid DSpark RMSNorm input");
    output.resize(u(rows) * u(width));
    cpu_pool().run(rows, [&](int row) {
        rms_norm_row(input + u(row) * u(width), weight, width,
                     epsilon, output.data() + u(row) * u(width));
    });
    return true;
}

bool dspark_hc_pre(const float * state, const float * fn_weight,
                   const float * base, const float scale[3], int tokens,
                   int n_embd, int n_hc, int sinkhorn_iterations,
                   float epsilon, std::vector<float> & working,
                   DsparkHcSplit & split, std::string * error) {
    working.clear();
    split = {};
    if (error) error->clear();
    const int hc_width = n_embd * n_hc;
    const int mix_width = 2 * n_hc + n_hc * n_hc;
    if (!state || !fn_weight || !base || !scale || tokens <= 0 ||
        n_embd <= 0 || n_hc <= 0 || n_hc > 8 || sinkhorn_iterations <= 0 ||
        epsilon < 0.0f || !finite_array(state, u(tokens) * u(hc_width)) ||
        !finite_array(fn_weight, u(mix_width) * u(hc_width)) ||
        !finite_array(base, u(mix_width)) ||
        !finite_array(scale, 3))
        return fail(error, "invalid DSpark HC-pre input");

    working.resize(u(tokens) * u(n_embd));
    split.values.resize(u(tokens) * u(mix_width));
    split.tokens = tokens;
    split.n_hc = n_hc;
    std::vector<float> normalized(u(tokens) * u(hc_width));
    std::vector<float> mix(u(tokens) * u(mix_width));
    cpu_pool().run(tokens, [&](int token) {
        rms_norm_row(state + u(token) * u(hc_width), nullptr, hc_width,
                     epsilon, normalized.data() + u(token) * u(hc_width));
    });
    cpu_pool().run(tokens * mix_width, [&](int task) {
            const int token = task / mix_width;
            const int row = task % mix_width;
            const float * matrix = fn_weight + u(row) * u(hc_width);
            const float * input = normalized.data() + u(token) * u(hc_width);
            float sum = 0.0f;
            for (int lane = 0; lane < hc_width; ++lane)
                sum += matrix[lane] * input[lane];
            mix[u(token) * u(mix_width) + u(row)] = sum;
    });
    cpu_pool().run(tokens, [&](int token) {
        const float * residual = state + u(token) * u(hc_width);
        const float * token_mix = mix.data() + u(token) * u(mix_width);
        float * values = split.values.data() + u(token) * u(mix_width);
        for (int h = 0; h < n_hc; ++h) {
            values[h] = sigmoid(token_mix[h] * scale[0] + base[h]) + kHcEpsilon;
            values[n_hc + h] = 2.0f * sigmoid(
                token_mix[n_hc + h] * scale[1] + base[n_hc + h]);
        }
        float * combine = values + 2 * n_hc;
        for (int destination = 0; destination < n_hc; ++destination) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (int source = 0; source < n_hc; ++source) {
                const int index = source + destination * n_hc;
                combine[index] = token_mix[2 * n_hc + index] * scale[2] +
                                 base[2 * n_hc + index];
                maximum = std::max(maximum, combine[index]);
            }
            float sum = 0.0f;
            for (int source = 0; source < n_hc; ++source) {
                const int index = source + destination * n_hc;
                combine[index] = std::exp(combine[index] - maximum);
                sum += combine[index];
            }
            for (int source = 0; source < n_hc; ++source) {
                const int index = source + destination * n_hc;
                combine[index] = combine[index] / sum + kHcEpsilon;
            }
        }
        for (int iteration = 0; iteration < sinkhorn_iterations; ++iteration) {
            for (int source = 0; source < n_hc; ++source) {
                float sum = 0.0f;
                for (int destination = 0; destination < n_hc; ++destination)
                    sum += combine[source + destination * n_hc];
                const float inverse = 1.0f / (sum + kHcEpsilon);
                for (int destination = 0; destination < n_hc; ++destination)
                    combine[source + destination * n_hc] *= inverse;
            }
            if (iteration + 1 == sinkhorn_iterations) break;
            for (int destination = 0; destination < n_hc; ++destination) {
                float sum = 0.0f;
                for (int source = 0; source < n_hc; ++source)
                    sum += combine[source + destination * n_hc];
                const float inverse = 1.0f / (sum + kHcEpsilon);
                for (int source = 0; source < n_hc; ++source)
                    combine[source + destination * n_hc] *= inverse;
            }
        }
        float * destination = working.data() + u(token) * u(n_embd);
        for (int lane = 0; lane < n_embd; ++lane) {
            float sum = 0.0f;
            for (int h = 0; h < n_hc; ++h)
                sum += values[h] * residual[u(h) * u(n_embd) + u(lane)];
            destination[lane] = sum;
        }
    });
    return true;
}

bool dspark_hc_post(const float * residual, const float * block_output,
                    const DsparkHcSplit & split, int n_embd,
                    std::vector<float> & output, std::string * error) {
    output.clear();
    if (error) error->clear();
    const int n_hc = split.n_hc;
    const int tokens = split.tokens;
    const int mix_width = 2 * n_hc + n_hc * n_hc;
    const int hc_width = n_embd * n_hc;
    if (!residual || !block_output || tokens <= 0 || n_embd <= 0 ||
        n_hc <= 0 || split.values.size() !=
            u(tokens) * u(mix_width))
        return fail(error, "invalid DSpark HC-post input");
    output.resize(u(tokens) * u(hc_width));
    cpu_pool().run(tokens * n_hc, [&](int task) {
        const int token = task / n_hc;
        const int destination = task % n_hc;
        const float * old = residual + u(token) * u(hc_width);
        const float * block = block_output + u(token) * u(n_embd);
        const float * values = split.values.data() + u(token) * u(mix_width);
        const float * post = values + n_hc;
        const float * combine = values + 2 * n_hc;
        float * next = output.data() + u(token) * u(hc_width);
        for (int lane = 0; lane < n_embd; ++lane) {
            float sum = block[lane] * post[destination];
            for (int source = 0; source < n_hc; ++source)
                sum += combine[destination + source * n_hc] *
                       old[u(source) * u(n_embd) + u(lane)];
            next[u(destination) * u(n_embd) + u(lane)] = sum;
        }
    });
    return true;
}

bool dspark_hc_out(const float * state, const float * fn_weight,
                   const float * base, float scale, int tokens, int n_embd,
                   int n_hc, float epsilon, std::vector<float> & output,
                   std::string * error) {
    output.clear();
    if (error) error->clear();
    const int hc_width = n_embd * n_hc;
    if (!state || !fn_weight || !base || tokens <= 0 || n_embd <= 0 ||
        n_hc <= 0 || n_hc > 8)
        return fail(error, "invalid DSpark HC-out input");
    output.resize(u(tokens) * u(n_embd));
    std::vector<float> normalized(u(tokens) * u(hc_width));
    std::vector<float> gates(u(tokens) * u(n_hc));
    cpu_pool().run(tokens, [&](int token) {
        const float * residual = state + u(token) * u(hc_width);
        rms_norm_row(residual, nullptr, hc_width, epsilon,
                     normalized.data() + u(token) * u(hc_width));
    });
    cpu_pool().run(tokens * n_hc, [&](int task) {
            const int token = task / n_hc;
            const int h = task % n_hc;
            const float * input = normalized.data() + u(token) * u(hc_width);
            const float * matrix = fn_weight + u(h) * u(hc_width);
            float mix = 0.0f;
            for (int lane = 0; lane < hc_width; ++lane)
                mix += matrix[lane] * input[lane];
            gates[u(token) * u(n_hc) + u(h)] =
                sigmoid(mix * scale + base[h]) + kHcEpsilon;
    });
    cpu_pool().run(tokens, [&](int token) {
        const float * residual = state + u(token) * u(hc_width);
        float * destination = output.data() + u(token) * u(n_embd);
        std::fill(destination, destination + n_embd, 0.0f);
        for (int h = 0; h < n_hc; ++h) {
            const float gate = gates[u(token) * u(n_hc) + u(h)];
            for (int lane = 0; lane < n_embd; ++lane)
                destination[lane] += gate *
                    residual[u(h) * u(n_embd) + u(lane)];
        }
    });
    return true;
}

bool dspark_route_topk(const float * normalized, const float * router_weight,
                       const float * selection_bias, int tokens, int n_embd,
                       int n_experts, int top_k, float expert_scale,
                       std::vector<int32_t> & selected,
                       std::vector<float> & weights, std::string * error,
                       std::vector<DsparkRouteBoundary> * boundaries) {
    selected.clear(); weights.clear();
    if (boundaries) boundaries->clear();
    if (error) error->clear();
    if (!normalized || !router_weight || tokens <= 0 || n_embd <= 0 ||
        n_experts <= 0 || top_k <= 0 || top_k > n_experts)
        return fail(error, "invalid DSpark router input");
    selected.resize(u(tokens) * u(top_k), -1);
    weights.resize(u(tokens) * u(top_k));
    if (boundaries) boundaries->resize(u(tokens));
    std::vector<float> probabilities(u(tokens) * u(n_experts));
    cpu_pool().run(tokens * n_experts, [&](int task) {
            const int token = task / n_experts;
            const int expert = task % n_experts;
            const float * row = normalized + u(token) * u(n_embd);
            const float * matrix = router_weight + u(expert) * u(n_embd);
            float logit = 0.0f;
            for (int lane = 0; lane < n_embd; ++lane) logit += matrix[lane] * row[lane];
            probabilities[u(token) * u(n_experts) + u(expert)] =
                std::sqrt(std::log1p(std::exp(-std::fabs(logit))) +
                          std::max(logit, 0.0f));
    });
    cpu_pool().run(tokens, [&](int token) {
        const float * token_probs = probabilities.data() + u(token) * u(n_experts);
        int32_t * ids = selected.data() + u(token) * u(top_k);
        for (int expert = 0; expert < n_experts; ++expert) {
            const float score = token_probs[expert] +
                (selection_bias ? selection_bias[expert] : 0.0f);
            for (int slot = 0; slot < top_k; ++slot) {
                const float current = ids[slot] < 0
                    ? -std::numeric_limits<float>::infinity()
                    : token_probs[ids[slot]] +
                      (selection_bias ? selection_bias[ids[slot]] : 0.0f);
                if (score > current) {
                    for (int move = top_k - 1; move > slot; --move)
                        ids[move] = ids[move - 1];
                    ids[slot] = expert;
                    break;
                }
            }
        }
        float sum = 0.0f;
        for (int slot = 0; slot < top_k; ++slot)
            sum += token_probs[ids[slot]];
        sum = std::max(sum, 6.103515625e-5f);
        for (int slot = 0; slot < top_k; ++slot)
            weights[u(token) * u(top_k) + u(slot)] =
                token_probs[ids[slot]] / sum * expert_scale;
        if (boundaries) {
            DsparkRouteBoundary boundary;
            boundary.selected_expert = ids[top_k - 1];
            const float selected_score = token_probs[boundary.selected_expert] +
                (selection_bias ? selection_bias[boundary.selected_expert] : 0.0f);
            float rejected_score = -std::numeric_limits<float>::infinity();
            for (int expert = 0; expert < n_experts; ++expert) {
                bool is_selected = false;
                for (int slot = 0; slot < top_k; ++slot) {
                    if (ids[slot] == expert) {
                        is_selected = true;
                        break;
                    }
                }
                if (is_selected) continue;
                const float score = token_probs[expert] +
                    (selection_bias ? selection_bias[expert] : 0.0f);
                if (score > rejected_score) {
                    rejected_score = score;
                    boundary.rejected_expert = expert;
                }
            }
            boundary.margin = selected_score - rejected_score;
            (*boundaries)[u(token)] = boundary;
        }
    });
    return true;
}

bool dspark_attention_reduce(const float * q, const float * kv,
                             const float * sinks,
                             const int32_t * query_positions,
                             const int32_t * kv_positions, int tokens,
                             int context_rows, int heads, int head_dim,
                             int rope_dims, float rope_base,
                             std::vector<float> & output,
                             std::string * error) {
    output.clear();
    if (error) error->clear();
    const int kv_rows = context_rows + tokens;
    if (!q || !kv || !query_positions || !kv_positions || tokens <= 0 ||
        context_rows < 0 || heads <= 0 || head_dim <= 0 || rope_dims <= 0 ||
        rope_dims > head_dim || (rope_dims & 1) || rope_base <= 0.0f)
        return fail(error, "invalid DSpark attention-reduction input");
    std::vector<float> rotated_q(q, q + u(tokens) * u(heads) * u(head_dim));
    std::vector<float> rotated_kv(kv, kv + u(kv_rows) * u(head_dim));
    cpu_pool().run(tokens * heads, [&](int task) {
        const int token = task / heads;
        rope_row(rotated_q.data() + u(task) * u(head_dim),
                 head_dim, rope_dims, query_positions[token], rope_base);
    });
    cpu_pool().run(kv_rows, [&](int row) {
        rope_row(rotated_kv.data() + u(row) * u(head_dim),
                 head_dim, rope_dims, kv_positions[row], rope_base);
    });
    output.resize(u(tokens) * u(heads) * u(head_dim));
    std::vector<float> all_scores(u(tokens) * u(heads) * u(kv_rows));
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    cpu_pool().run(tokens * heads, [&](int task) {
            const int token = task / heads;
            const int head = task % heads;
            float * scores = all_scores.data() + u(task) * u(kv_rows);
            const float * query = rotated_q.data() +
                (u(token) * u(heads) + u(head)) * u(head_dim);
            float maximum = sinks ? sinks[head] :
                -std::numeric_limits<float>::infinity();
            for (int row = 0; row < kv_rows; ++row) {
                const float * key = rotated_kv.data() + u(row) * u(head_dim);
                float dot = 0.0f;
                for (int lane = 0; lane < head_dim; ++lane) dot += query[lane] * key[lane];
                scores[row] = dot * attention_scale;
                maximum = std::max(maximum, scores[row]);
            }
            float denominator = sinks ? std::exp(sinks[head] - maximum) : 0.0f;
            for (int row = 0; row < kv_rows; ++row) {
                scores[row] = std::exp(scores[row] - maximum);
                denominator += scores[row];
            }
            float * result = output.data() +
                (u(token) * u(heads) + u(head)) * u(head_dim);
            std::fill(result, result + head_dim, 0.0f);
            for (int row = 0; row < kv_rows; ++row) {
                const float probability = scores[row] / denominator;
                const float * value = rotated_kv.data() + u(row) * u(head_dim);
                for (int lane = 0; lane < head_dim; ++lane)
                    result[lane] += probability * value[lane];
            }
            rope_row(result, head_dim, rope_dims, -query_positions[token], rope_base);
    });
    return true;
}

}  // namespace ember::xdna2
