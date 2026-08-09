#include <stdio.h>

#include "../src/server/background_gate.h"

static int g_pass, g_fail;
#define CHECK(c, m) do { if (c) g_pass++; else { g_fail++; printf("  FAIL: %s\n", m); } } while (0)

int main(void) {
    ember_background_gate g;
    ember_background_gate_init(&g, 5.0, 60.0);

    CHECK(ember_background_gate_ready(&g, 10.0, 10.0),
          "background runs immediately before foreground activity");

    ember_background_gate_note_foreground(&g, 20.0);
    CHECK(!ember_background_gate_ready(&g, 21.0, 24.9),
          "background waits through foreground idle window");
    CHECK(ember_background_gate_ready(&g, 21.0, 25.0),
          "background runs after idle window");

    ember_background_gate_note_foreground(&g, 100.0);
    CHECK(ember_background_gate_ready_at(&g, 10.0) == 70.0,
          "maximum wait bounds repeated foreground deferral");
    CHECK(ember_background_gate_ready(&g, 10.0, 70.0),
          "maximum-wait deadline is eligible");

    ember_background_gate_init(&g, 0.0, 0.0);
    ember_background_gate_note_foreground(&g, 100.0);
    CHECK(ember_background_gate_ready(&g, 101.0, 101.0),
          "zero idle disables deferral");

    printf("ember background gate tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
