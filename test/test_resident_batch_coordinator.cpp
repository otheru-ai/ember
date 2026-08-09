#include "resident_batch_coordinator.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>

using namespace dflash::common;

static int passed = 0;
static int failed = 0;

#define CHECK(expr) do { \
    if (expr) { ++passed; } \
    else { ++failed; std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                                  __FILE__, __LINE__, #expr); } \
} while (0)

struct FakeResidentBackend : ResidentBatchBackend {
    struct Session {
        GenerateRequest request;
        int prefilled = 0;
        int generated = 0;
        bool ready = false;
        bool terminal = false;
        bool cancelled = false;
        bool failed = false;
    };

    std::unordered_map<ContinuousBatchSessionId, Session> sessions;
    std::vector<int> decode_batch_sizes;
    int destroyed = 0;
    int restore_position = 2;
    bool fail_decode = false;

    bool resident_session_create(
            ContinuousBatchSessionId id,
            const GenerateRequest &request,
            const DaemonIO &,
            int restore_slot,
            std::string *error) override {
        if (sessions.count(id)) {
            if (error) *error = "duplicate";
            return false;
        }
        Session session;
        session.request = request;
        session.prefilled = restore_slot >= 0 ? restore_position : 0;
        if (session.prefilled > (int)request.prompt.size()) {
            if (error) *error = "bad restore";
            return false;
        }
        session.ready =
            session.prefilled == (int)request.prompt.size() &&
            request.n_gen > 0;
        session.terminal = request.n_gen == 0 &&
                           session.prefilled == (int)request.prompt.size();
        sessions.emplace(id, std::move(session));
        return true;
    }

    bool resident_session_destroy(ContinuousBatchSessionId id) override {
        if (sessions.erase(id) != 1) return false;
        ++destroyed;
        return true;
    }

    bool resident_session_cancel(ContinuousBatchSessionId id) override {
        auto it = sessions.find(id);
        if (it == sessions.end() || it->second.terminal ||
            it->second.cancelled || it->second.failed)
            return false;
        it->second.cancelled = true;
        it->second.terminal = true;
        it->second.ready = false;
        return true;
    }

    bool resident_session_decode_ready(
            ContinuousBatchSessionId id) const override {
        auto it = sessions.find(id);
        return it != sessions.end() && it->second.ready;
    }

    SessionStatus resident_session_status(
            ContinuousBatchSessionId id) const override {
        SessionStatus status;
        auto it = sessions.find(id);
        if (it == sessions.end()) {
            status.failed = true;
            return status;
        }
        status.prefilled_tokens = it->second.prefilled;
        status.decode_ready = it->second.ready;
        status.terminal = it->second.terminal;
        status.cancelled = it->second.cancelled;
        status.failed = it->second.failed;
        return status;
    }

    GenerateResult resident_session_result(
            ContinuousBatchSessionId id) const override {
        GenerateResult result;
        auto it = sessions.find(id);
        if (it == sessions.end() || it->second.failed) {
            result.fail(GenerateErrorCode::BackendSpecific);
            return result;
        }
        result.succeed();
        result.tokens.assign((size_t)it->second.generated, 42);
        return result;
    }

    bool resident_session_snapshot(
            ContinuousBatchSessionId id, int slot) override {
        return sessions.count(id) != 0 && slot >= 0;
    }

    ContinuousBatchPrefillCompletion prefill(
            ContinuousBatchSessionId id, int requested) override {
        auto it = sessions.find(id);
        if (it == sessions.end()) return {};
        Session &session = it->second;
        const int consumed = std::min(
            requested,
            (int)session.request.prompt.size() - session.prefilled);
        if (consumed <= 0) return {};
        session.prefilled += consumed;
        if (session.prefilled == (int)session.request.prompt.size()) {
            if (session.request.n_gen > 0) session.ready = true;
            else session.terminal = true;
        }
        return {true, consumed};
    }

    std::vector<ContinuousBatchDecodeCompletion> decode_batch(
            const std::vector<ContinuousBatchSessionId> &ids) override {
        decode_batch_sizes.push_back((int)ids.size());
        std::vector<ContinuousBatchDecodeCompletion> result;
        if (fail_decode) {
            for (ContinuousBatchSessionId id : ids)
                result.push_back({id, false, false});
            return result;
        }
        for (ContinuousBatchSessionId id : ids) {
            auto it = sessions.find(id);
            if (it == sessions.end() || !it->second.ready) {
                result.push_back({id, false, false});
                continue;
            }
            Session &session = it->second;
            session.ready = false;
            ++session.generated;
            session.terminal =
                session.generated >= session.request.n_gen;
            if (!session.terminal) session.ready = true;
            result.push_back({id, true, session.terminal});
        }
        return result;
    }
};

static GenerateRequest request(int prompt_tokens, int generated_tokens) {
    GenerateRequest req;
    req.prompt.resize((size_t)prompt_tokens, 7);
    req.n_gen = generated_tokens;
    return req;
}

int main() {
    {
        FakeResidentBackend backend;
        ResidentBatchCoordinator coordinator(
            backend, ContinuousBatchConfig{3, 4, 2, 0});
        std::string error;
        auto a = coordinator.admit(request(0, 3), {}, -1, 0, &error);
        auto b = coordinator.admit(request(0, 2), {}, -1, 0, &error);
        auto c = coordinator.admit(request(0, 1), {}, -1, 0, &error);
        CHECK(a.has_value());
        CHECK(b.has_value());
        CHECK(c.has_value());
        CHECK(!coordinator.admit(request(1, 1), {}, -1, 0, &error));

        for (int i = 0; i < 30 &&
                        (!coordinator.terminal(*a) ||
                         !coordinator.terminal(*b) ||
                         !coordinator.terminal(*c)); ++i) {
            coordinator.pump(i);
        }
        CHECK(coordinator.terminal(*a));
        CHECK(coordinator.terminal(*b));
        CHECK(coordinator.terminal(*c));
        CHECK(coordinator.result(*a)->tokens.size() == 3);
        CHECK(coordinator.result(*b)->tokens.size() == 2);
        CHECK(coordinator.result(*c)->tokens.size() == 1);
        CHECK(std::find(backend.decode_batch_sizes.begin(),
                        backend.decode_batch_sizes.end(), 3) !=
              backend.decode_batch_sizes.end());
        CHECK(coordinator.release(*a));
        CHECK(coordinator.release(*b));
        CHECK(coordinator.release(*c));
        CHECK(backend.destroyed == 3);
    }

    {
        FakeResidentBackend backend;
        ResidentBatchCoordinator coordinator(
            backend, ContinuousBatchConfig{2, 4, 2, 0});
        std::string error;
        auto restored = coordinator.admit(
            request(5, 1), {}, 4, backend.restore_position, &error);
        CHECK(restored.has_value());
        CHECK(backend.sessions[*restored].prefilled == 2);
        coordinator.pump(0);
        CHECK(backend.sessions[*restored].prefilled == 5);
        coordinator.pump(1);
        CHECK(coordinator.terminal(*restored));
    }

    {
        FakeResidentBackend backend;
        ResidentBatchCoordinator coordinator(
            backend, ContinuousBatchConfig{1, 4, 2, 0});
        std::string error;
        CHECK(!coordinator.admit(request(1, 1), {}, -1, 2, &error));
        CHECK(!error.empty());
        auto id = coordinator.admit(request(0, 3), {}, -1, 0, &error);
        CHECK(id.has_value());
        CHECK(coordinator.cancel(*id));
        CHECK(backend.sessions[*id].cancelled);
        CHECK(coordinator.terminal(*id));
        CHECK(coordinator.release(*id));
    }

    {
        FakeResidentBackend backend;
        backend.fail_decode = true;
        ResidentBatchCoordinator coordinator(
            backend, ContinuousBatchConfig{1, 4, 2, 0});
        std::string error;
        auto id = coordinator.admit(request(0, 1), {}, -1, 0, &error);
        CHECK(id.has_value());
        coordinator.pump(0);
        CHECK(coordinator.terminal(*id));
        auto failed_result = coordinator.result(*id);
        CHECK(failed_result.has_value() && !failed_result->ok());
        CHECK(failed_result &&
              failed_result->error_code() == "backend_specific");
    }

    std::printf("resident batch coordinator tests: %d passed, %d failed\n",
                passed, failed);
    return failed == 0 ? 0 : 1;
}
