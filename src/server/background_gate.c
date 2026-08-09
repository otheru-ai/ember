#include "background_gate.h"

#include <stddef.h>

void ember_background_gate_init(ember_background_gate *g,
                                double idle_secs, double max_wait_secs) {
    if (!g) return;
    g->idle_secs = idle_secs > 0.0 ? idle_secs : 0.0;
    g->max_wait_secs = max_wait_secs > 0.0 ? max_wait_secs : 0.0;
    g->last_foreground_at = 0.0;
}

void ember_background_gate_note_foreground(ember_background_gate *g, double now) {
    if (g && now > g->last_foreground_at) g->last_foreground_at = now;
}

double ember_background_gate_ready_at(const ember_background_gate *g,
                                      double enqueued_at) {
    if (!g) return enqueued_at;
    double idle_at = g->last_foreground_at > 0.0
                         ? g->last_foreground_at + g->idle_secs
                         : enqueued_at;
    if (g->max_wait_secs <= 0.0) return idle_at;
    double forced_at = enqueued_at + g->max_wait_secs;
    return forced_at < idle_at ? forced_at : idle_at;
}

bool ember_background_gate_ready(const ember_background_gate *g,
                                 double enqueued_at, double now) {
    return now >= ember_background_gate_ready_at(g, enqueued_at);
}
