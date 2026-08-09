#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../src/model/model_card.h"
static int g_pass=0,g_fail=0;
#define CHECK(c,m) do{ if(c)g_pass++; else{g_fail++;printf("  FAIL: %s\n",m);} }while(0)
int main(void){
    printf("ember model_card tests\n");
    ember_model_card c;
    ember_model_card_load(&c, NULL);  // defaults
    CHECK(c.hard_limit_reply_budget==1024, "default reply budget 1024");
    CHECK(c.thinking_terminator_hint && strstr(c.thinking_terminator_hint,"</think>"),
          "terminator hint present + closes think");
    CHECK(c.temperature == 0.6 && c.top_p == 0.95 && c.top_k == 40 &&
          c.min_p == 0.0 && c.presence_penalty == 0.0 &&
          c.repetition_penalty == 1.0,
          "complete DeepSeek sampler defaults");
    // The empty-content bug: at max_tokens=600 < hard_limit → think budget 0
    CHECK(ember_model_card_think_budget(&c,"high",600)==0,
          "tiny budget disables thinking (no reply-starvation)");
    // Realistic: max_tokens=4096, high tier=16384 → clamped to 4096-1024=3072
    CHECK(ember_model_card_think_budget(&c,"high",4096)==3072,
          "think budget clamped to reply room");
    // low effort under a big client budget → the low tier caps it
    CHECK(ember_model_card_think_budget(&c,"low",65536)==4096, "low tier caps");
    // unset effort defaults to high
    CHECK(ember_model_card_think_budget(&c,NULL,4096)==3072, "default effort=high");

    // ── reasoning-effort wiring contract (src/server/main.c) ──
    // main.c turns a think budget T into the backend's reply reserve as
    // (max_tokens - T), because the force-close fires at (max_tokens - reserve).
    // These assert the two properties that wiring depends on.
    #define RESERVE(effort, mt) \
        ((ember_model_card_think_budget(&c,(effort),(mt)) > 0 && \
          (mt) - ember_model_card_think_budget(&c,(effort),(mt)) > c.hard_limit_reply_budget) \
             ? (mt) - ember_model_card_think_budget(&c,(effort),(mt)) \
             : c.hard_limit_reply_budget)

    // 1. BACKWARD COMPATIBILITY: when the tier is looser than the room left by
    //    max_tokens, the clamp makes T == room and the reserve must come out
    //    exactly hard_limit_reply_budget — i.e. identical to the pre-tier build.
    //    This is production's case (agent gateway: effort=high, max_tokens=4096).
    CHECK(RESERVE("high",4096)==1024, "loose tier -> reserve unchanged (pre-tier parity)");
    CHECK(RESERVE("low",4096)==1024,  "tier above room cannot shrink the think cap");

    // 2. THE TIER BITES when it is the binding constraint: low=4096 under a
    //    16384 budget must stop thinking at 4096, i.e. reserve 16384-4096.
    CHECK(ember_model_card_think_budget(&c,"low",16384)==4096, "low tier binds at 16k");
    CHECK(RESERVE("low",16384)==12288, "binding tier -> reserve = max_tokens - tier");
    CHECK(RESERVE("medium",16384)==8192, "medium tier -> reserve 16384-8192");
    // high tier (16384) == max_tokens leaves no room, so the clamp wins again.
    CHECK(RESERVE("high",16384)==1024, "high tier at 16k falls back to reply room");

    // 3. The reserve is monotone in effort: more effort never reserves more
    //    (i.e. never thinks less). Guards against an inverted tier table.
    CHECK(RESERVE("low",16384) >= RESERVE("medium",16384), "low reserves >= medium");
    CHECK(RESERVE("medium",16384) >= RESERVE("high",16384), "medium reserves >= high");

    // 4. The reply guarantee survives every tier: reserve never drops below the
    //    card's hard limit, so the force-close always leaves room to answer.
    CHECK(RESERVE("low",16384) >= c.hard_limit_reply_budget &&
          RESERVE("high",4096) >= c.hard_limit_reply_budget,
          "reserve never below hard_limit_reply_budget");
    #undef RESERVE

    // 5. Both x-high spellings resolve to the SAME tier. The card JSON and
    //    OpenAI use "x-high"; chat_api.c's effort_to_mode matched "xhigh".
    //    If these disagree a request silently gets one effort's think mode and
    //    another's token budget.
    CHECK(ember_model_card_think_budget(&c,"x-high",65536) ==
          ember_model_card_think_budget(&c,"xhigh",65536),
          "x-high and xhigh resolve to the same tier");
    CHECK(ember_model_card_think_budget(&c,"x-high",65536)==24576, "x-high tier is 24576");
    // and an unknown effort still falls back to the high tier (lenient)
    CHECK(ember_model_card_think_budget(&c,"bogus",65536)==
          ember_model_card_think_budget(&c,"high",65536),
          "unknown effort falls back to high tier");

    ember_model_card_free(&c);

    char path[] = "/tmp/ember-card-test-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "create model-card fixture");
    if (fd >= 0) {
        const char bad[] =
            "{\"max_tokens\":1e20,\"hard_limit_reply_budget\":-1,"
            "\"reasoning_effort_tiers\":{\"low\":-2,\"high\":1e20},"
            "\"sampling\":{\"temperature\":99,\"top_p\":2,"
            "\"top_k\":2.5,\"min_p\":-1,\"presence_penalty\":3,"
            "\"repetition_penalty\":0}}";
        CHECK(write(fd, bad, sizeof(bad)-1) == (ssize_t)(sizeof(bad)-1),
              "write model-card fixture");
        close(fd);
        CHECK(ember_model_card_load(&c, path), "load bounded model card");
        CHECK(c.max_tokens == 16384 &&
              c.hard_limit_reply_budget == 1024,
              "out-of-range token budgets keep safe defaults");
        CHECK(c.tiers.low == 4096 && c.tiers.high == 16384,
              "out-of-range reasoning tiers keep safe defaults");
        CHECK(c.temperature == 0.6 && c.top_p == 0.95 && c.top_k == 40 &&
              c.min_p == 0.0 && c.presence_penalty == 0.0 &&
              c.repetition_penalty == 1.0,
              "out-of-range sampling values keep safe defaults");
        ember_model_card_free(&c);
        unlink(path);
    }

    char sampling_path[] = "/tmp/ember-card-sampling-test-XXXXXX";
    fd = mkstemp(sampling_path);
    CHECK(fd >= 0, "create sampling-card fixture");
    if (fd >= 0) {
        const char sampling[] =
            "{\"sampling\":{\"temperature\":0.7,\"top_p\":0.9,"
            "\"top_k\":17,\"min_p\":0.03,\"presence_penalty\":-0.4,"
            "\"repetition_penalty\":1.15}}";
        CHECK(write(fd, sampling, sizeof(sampling)-1) ==
                  (ssize_t)(sizeof(sampling)-1),
              "write sampling-card fixture");
        close(fd);
        CHECK(ember_model_card_load(&c, sampling_path),
              "load complete sampling card");
        CHECK(c.temperature == 0.7 && c.top_p == 0.9 && c.top_k == 17 &&
              c.min_p == 0.03 && c.presence_penalty == -0.4 &&
              c.repetition_penalty == 1.15,
              "all card sampler fields are retained");
        ember_model_card_free(&c);
        unlink(sampling_path);
    }
    printf("──────────────────────────────\n  %d passed, %d failed\n",g_pass,g_fail);
    return g_fail?1:0;
}
