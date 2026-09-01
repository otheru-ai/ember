#include <stdio.h>
#include <string.h>
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
    ember_model_card_free(&c);
    printf("──────────────────────────────\n  %d passed, %d failed\n",g_pass,g_fail);
    return g_fail?1:0;
}
