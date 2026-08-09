// Sanitized degraded-output fixtures. These retain parser and SSE coverage
// without embedding production commands, internal hostnames, or credentials.
#pragma once

static const char REAL_PSEUDO_MARKER_1297[] =
    " <?DSML?tool_caddy>ordinary malformed marker text";
static const char REAL_PSEUDO_MARKER_136[] =
    "<?DSML?tool_alls><invoke name= \"skill_manage\">text</?DSML?_manage>";
static const char REAL_FRAGMENTATION_130[] =
    "?DSML?tool_cards> KEY=\"redacted\" \"-\" \"fixture\"";
static const char REAL_REPETITION_81[] =
    "The canonical charset is repeated. The canonical charset is repeated.";
