#ifndef EMBER_BACKGROUND_GATE_H
#define EMBER_BACKGROUND_GATE_H

#include <stdbool.h>

// Scheduling policy for low-priority maintenance requests. Times use the same
// monotonic clock and are expressed in seconds.
typedef struct {
    double idle_secs;
    double max_wait_secs;
    double last_foreground_at;
} ember_background_gate;

void ember_background_gate_init(ember_background_gate *g,
                                double idle_secs, double max_wait_secs);
void ember_background_gate_note_foreground(ember_background_gate *g, double now);
bool ember_background_gate_ready(const ember_background_gate *g,
                                 double enqueued_at, double now);
double ember_background_gate_ready_at(const ember_background_gate *g,
                                      double enqueued_at);

#endif
