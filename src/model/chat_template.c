#include "chat_template.h"
#include "chat_template_qwen4.h"

#include <stdlib.h>
#include <string.h>

#include "../common/buf.h"
#include "../common/json.h"

// Concatenated pieces isolate each \x escape so it can't eat a following hex
// digit (e.g. "\x9cbegin" would parse \x9cb, out of range).
#define PIPE   "\xef\xbd\x9c"   // U+FF5C ｜
#define USCORE "\xe2\x96\x81"   // U+2581 ▁
#define BOS  "<" PIPE "begin" USCORE "of" USCORE "sentence" PIPE ">"
#define EOS  "<" PIPE "end" USCORE "of" USCORE "sentence" PIPE ">"
#define USER "<" PIPE "User" PIPE ">"
#define ASST "<" PIPE "Assistant" PIPE ">"

// ds4 DS4_REASONING_EFFORT_MAX_PREFIX (byte-exact) — injected after BOS on MAX.
static const char *THINK_MAX_PREFIX =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

// ds4 role_is_system / role_is_user_like.
static bool role_is_system(const char *r) {
    return r && (!strcmp(r, "system") || !strcmp(r, "developer"));
}
static bool role_is_user_like(const char *r) {
    return r && (!strcmp(r, "user") || !strcmp(r, "tool") || !strcmp(r, "function"));
}

// ds4 chat_history_uses_tool_context: tools active, or any tool result / prior
// assistant tool-call in history.
static bool uses_tool_context(const ember_chat_request *req) {
    if (req->has_tools) return true;
    for (int i = 0; i < req->n_messages; i++) {
        const ember_chat_msg *m = &req->messages[i];
        if (m->role &&
            ((!strcmp(m->role, "assistant") && m->calls.len > 0) ||
             !strcmp(m->role, "tool") || !strcmp(m->role, "function")))
            return true;
    }
    return false;
}

// ds4 append_tool_result_text: keep '<','>','&' literal; only rewrite the exact
// closing sentinel </tool_result> to &lt; so tool output can't close the wrapper.
static void append_tool_result_text(ember_buf *b, const char *s) {
    const char *end = "</tool_result>";
    size_t endlen = 14;
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) { ember_buf_puts(b, "&lt;"); s++; }
        else ember_buf_putc(b, *s++);
    }
}

// Minimal DSML attribute escape (tool/param names). Mirrors ds4_server.c
// append_dsml_attr_escaped (ds4_server.c:2193-2199): all four of ", &, <, >
// are entity-escaped. This is attribute (name) escaping only — string *values*
// deliberately keep '>' literal per the tool preamble, so the '>' case here
// does not conflict with that. A tool/param name containing '>' would otherwise
// render an unescaped '>' that ds4 escapes, breaking byte-parity of the turn.
static void append_attr_escaped(ember_buf *b, const char *s) {
    for (s = s ? s : ""; *s; s++) {
        if (*s == '"') ember_buf_puts(b, "&quot;");
        else if (*s == '&') ember_buf_puts(b, "&amp;");
        else if (*s == '<') ember_buf_puts(b, "&lt;");
        else if (*s == '>') ember_buf_puts(b, "&gt;");
        else ember_buf_putc(b, *s);
    }
}

// B#2: a string argument value containing the exact closing-parameter sentinel
// would otherwise terminate its <parameter> early and rewrite the prompt
// structure. ds4 append_dsml_parameter_text: rewrite the sentinel's leading '<'
// to "&lt;" (the same escape the preamble instructs the model to produce, so a
// later parse unescapes it straight back to the original value).
static void append_dsml_param_text(ember_buf *b, const char *s) {
    const char *end = "</" PIPE "DSML" PIPE "parameter>";
    size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) { ember_buf_puts(b, "&lt;"); s++; }
        else ember_buf_putc(b, *s++);
    }
}

// B#2: same protection for JSON-literal (string="false") values, where the
// sentinel can hide inside a nested JSON string. ds4 append_dsml_json_literal:
// rewrite the sentinel's '<' to its JSON unicode escape (backslash-u-003c) so
// the literal stays valid JSON and round-trips to the same value, but no raw
// sentinel survives to break the DSML wrapper.
static void append_dsml_json_literal(ember_buf *b, const char *s) {
    const char *end = "</" PIPE "DSML" PIPE "parameter>";
    size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) { ember_buf_puts(b, "\\u003c"); s++; }
        else ember_buf_putc(b, *s++);
    }
}

// Render one call's JSON arguments as DSML <parameter> tags (ds4
// append_dsml_arguments_from_json equivalent). string="true" for JSON strings
// (raw text), string="false" for everything else (verbatim JSON). Returns false
// when args_json is not a JSON object, so the caller can emit a raw-arguments
// fallback instead of an empty call (B#2).
static bool append_dsml_args(ember_buf *b, const char *args_json) {
    ember_json *o = ember_json_parse(args_json);
    bool is_object = o && o->type == EMBER_JSON_OBJECT;
    if (is_object) {
        for (int i = 0; i < ember_json_len(o); i++) {
            const char *key = ember_json_key(o, i);
            const ember_json *v = ember_json_at(o, i);
            bool is_str = v && v->type == EMBER_JSON_STRING;
            ember_buf_puts(b, "<" PIPE "DSML" PIPE "parameter name=\"");
            append_attr_escaped(b, key ? key : "");
            ember_buf_puts(b, is_str ? "\" string=\"true\">" : "\" string=\"false\">");
            if (is_str) {
                append_dsml_param_text(b, ember_json_str(v, ""));  // B#2: escape sentinel
            } else {
                char *d = ember_json_dump(v);
                append_dsml_json_literal(b, d ? d : "null");       // B#2: escape sentinel
                free(d);
            }
            ember_buf_puts(b, "</" PIPE "DSML" PIPE "parameter>\n");
        }
    }
    if (o) ember_json_free(o);
    return is_object;
}

// ds4 append_dsml_tool_calls_text: render an assistant turn's tool calls.
static void append_dsml_tool_calls(ember_buf *b, const ember_tool_calls *calls) {
    if (!calls || calls->len == 0) return;
    ember_buf_puts(b, "\n\n<" PIPE "DSML" PIPE "tool_calls>\n");
    for (int i = 0; i < calls->len; i++) {
        ember_buf_puts(b, "<" PIPE "DSML" PIPE "invoke name=\"");
        append_attr_escaped(b, calls->calls[i].name);
        ember_buf_puts(b, "\">\n");
        // B#2: arguments that aren't a JSON object (array-typed or malformed
        // strings accepted in request history) would otherwise render as an
        // EMPTY call. Mirror ds4 append_dsml_tool_calls_text: emit a single
        // raw-arguments parameter so the call survives instead of being dropped.
        if (!append_dsml_args(b, calls->calls[i].arguments)) {
            ember_buf_puts(b, "<" PIPE "DSML" PIPE "parameter name=\"arguments\" string=\"true\">");
            append_dsml_param_text(b, calls->calls[i].arguments);
            ember_buf_puts(b, "</" PIPE "DSML" PIPE "parameter>\n");
        }
        ember_buf_puts(b, "</" PIPE "DSML" PIPE "invoke>\n");
    }
    ember_buf_puts(b, "</" PIPE "DSML" PIPE "tool_calls>");
}

// ds4 append_tools_prompt_text (byte-exact) + the raw schemas + trailer.
static void append_tool_preamble(ember_buf *b, const char *schemas) {
    if (!schemas || !schemas[0]) return;
    ember_buf_puts(b,
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user question. "
        "You can invoke tools by writing a \"<" PIPE "DSML" PIPE "tool_calls>\" block like the following:\n\n"
        "<" PIPE "DSML" PIPE "tool_calls>\n"
        "<" PIPE "DSML" PIPE "invoke name=\"$TOOL_NAME\">\n"
        "<" PIPE "DSML" PIPE "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" PIPE "DSML" PIPE "parameter>\n"
        "...\n"
        "</" PIPE "DSML" PIPE "invoke>\n"
        "<" PIPE "DSML" PIPE "invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</" PIPE "DSML" PIPE "invoke>\n"
        "</" PIPE "DSML" PIPE "tool_calls>\n\n"
        "String parameters should be specified as raw text and set `string=\"true\"`. "
        "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. "
        "Only if a string value itself contains the exact closing parameter tag `</" PIPE "DSML" PIPE "parameter>`, write that tag as `&lt;/" PIPE "DSML" PIPE "parameter>` inside the value. "
        "If a string value contains any other literal DSML control tag, escape that tag's leading `<` as `&lt;`; unescaped nested DSML is rejected. "
        "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n");
    ember_buf_puts(b, schemas);
    ember_buf_puts(b,
        "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. "
        "Use the exact parameter names from the schemas.");
}

// Build the tool-schemas block: each tool's inner `function` object, joined by a
// single '\n', no trailing newline (ds4 append_raw_json_line spacing).
static char *build_schemas(const char *tools_json) {
    ember_json *tools = ember_json_parse(tools_json);
    ember_buf b = {0};
    if (tools && tools->type == EMBER_JSON_ARRAY) {
        for (int i = 0; i < ember_json_len(tools); i++) {
            const ember_json *t = ember_json_at(tools, i);
            const ember_json *fn = ember_json_get(t, "function");
            char *dump = ember_json_dump(fn ? fn : t);
            if (i) ember_buf_putc(&b, '\n');
            ember_buf_puts(&b, dump ? dump : "");
            free(dump);
        }
    } else if (tools_json) {
        ember_buf_puts(&b, tools_json);
    }
    if (tools) ember_json_free(tools);
    char *s = ember_buf_take(&b);
    return s ? s : strdup("");
}

char *ember_render_prompt(const ember_chat_request *req, bool enable_thinking,
                          ember_think_mode think_mode, bool add_generation_prompt) {
    if (!req || req->n_messages < 0 ||
        (req->n_messages > 0 && !req->messages)) return NULL;
    if (req->prompt_profile == EMBER_PROMPT_QWEN4_CHATML) {
        return ember_qwen4_render_prompt(req, enable_thinking,
                                         add_generation_prompt);
    }
    const bool think = enable_thinking;
    const bool tool_context = uses_tool_context(req);
    int last_user_idx = -1;
    for (int i = 0; i < req->n_messages; i++)
        if (role_is_user_like(req->messages[i].role)) last_user_idx = i;

    // DeepSeek-V4 reference / current Unsloth template order: client system
    // messages first, then the tool preamble and schemas. The ordering is part
    // of the model's trained prompt ABI; changing it for cache convenience made
    // Ember's prompts diverge from the golden renderer.
    //
    // DELIBERATE DIVERGENCE FROM ds4: ds4_server.c:2438 renders tool schemas
    // *before* the client system content (its own test:
    // test_render_chat_prompt_text_renders_tools_before_system). Ember does the
    // opposite to match the DeepSeek-V4/Unsloth trained template. This order is
    // pinned by test_reference_system_then_tools_order (test_chat_template.c).
    ember_buf sys = {0};
    for (int i = 0; i < req->n_messages; i++) {
        if (!role_is_system(req->messages[i].role)) continue;
        if (sys.len) ember_buf_puts(&sys, "\n\n");
        ember_buf_puts(&sys, req->messages[i].content ? req->messages[i].content : "");
    }
    if (req->has_tools) {
        if (sys.len) ember_buf_puts(&sys, "\n\n");
        char *schemas = build_schemas(req->tools_json);
        append_tool_preamble(&sys, schemas);
        free(schemas);
        if (req->tool_choice_required)
            ember_buf_puts(&sys,
                "\n\nYou MUST invoke at least one available tool in this "
                "assistant turn; a text-only answer is invalid.");
        if (req->n_tool_choice_names > 0) {
            ember_buf_puts(&sys,
                "\nOnly the following tool names may be invoked in this turn: ");
            for (int i = 0; i < req->n_tool_choice_names; ++i) {
                if (i) ember_buf_puts(&sys, ", ");
                ember_buf_puts(&sys, req->tool_choice_names[i]);
            }
            ember_buf_putc(&sys, '.');
        }
        if (!req->parallel_tool_calls)
            ember_buf_puts(&sys,
                "\nInvoke at most one tool in this assistant turn.");
    }

    ember_buf b = {0};
    ember_buf_puts(&b, BOS);
    if (think && think_mode == EMBER_THINK_MAX) ember_buf_puts(&b, THINK_MAX_PREFIX);
    ember_buf_append(&b, sys.ptr ? sys.ptr : "", sys.len);
    ember_buf_free(&sys);

    bool pending_assistant = false;
    bool pending_tool_result = false;
    for (int i = 0; i < req->n_messages; i++) {
        const ember_chat_msg *m = &req->messages[i];
        const char *content = m->content ? m->content : "";
        if (role_is_system(m->role)) {
            continue;
        } else if (m->role && !strcmp(m->role, "user")) {
            ember_buf_puts(&b, USER);
            ember_buf_puts(&b, content);
            pending_assistant = true;
            pending_tool_result = false;
        } else if (m->role &&
                   (!strcmp(m->role, "tool") || !strcmp(m->role, "function"))) {
            if (!pending_tool_result) ember_buf_puts(&b, USER);
            ember_buf_puts(&b, "<tool_result>");
            append_tool_result_text(&b, content);
            ember_buf_puts(&b, "</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if (m->role && !strcmp(m->role, "assistant")) {
            if (pending_assistant) {
                ember_buf_puts(&b, ASST);
                if (think) {
                    if (tool_context || i > last_user_idx) {
                        ember_buf_puts(&b, "<think>");
                        ember_buf_puts(&b, m->reasoning ? m->reasoning : "");
                        ember_buf_puts(&b, "</think>");
                    } else {
                        ember_buf_puts(&b, "</think>");
                    }
                } else {
                    ember_buf_puts(&b, "</think>");
                }
            }
            ember_buf_puts(&b, content);
            if (m->raw_tool_text) {
                // Dwarfstar-compatible replay scope: only the exact sampled
                // DSML block is substituted. Reasoning and visible content are
                // always rendered through the current template, so replay can
                // never smuggle a prior hidden trajectory into this request.
                ember_buf_puts(&b, "\n\n");
                ember_buf_puts(&b, m->raw_tool_text);
            } else {
                append_dsml_tool_calls(&b, &m->calls);
            }
            ember_buf_puts(&b, EOS);
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if (add_generation_prompt && pending_assistant) {
        ember_buf_puts(&b, ASST);
        ember_buf_puts(&b, think ? "<think>" : "</think>");
    }

    char *s = ember_buf_take(&b);
    return s ? s : strdup("");
}

char *ember_render_tool_continuation_suffix(const ember_chat_request *req,
                                            bool enable_thinking) {
    if (!req || req->n_messages <= 0 || !req->messages) return NULL;
    if (req->prompt_profile == EMBER_PROMPT_QWEN4_CHATML) {
        return ember_qwen4_render_tool_continuation_suffix(req,
                                                           enable_thinking);
    }
    ember_buf b = {0};
    ember_buf_puts(&b, EOS);
    bool opened_user = false;
    for (int i = 0; i < req->n_messages; ++i) {
        const ember_chat_msg *m = &req->messages[i];
        if (!m->role ||
            (strcmp(m->role, "tool") != 0 &&
             strcmp(m->role, "function") != 0)) {
            ember_buf_free(&b);
            return NULL;
        }
        if (!opened_user) {
            ember_buf_puts(&b, USER);
            opened_user = true;
        }
        ember_buf_puts(&b, "<tool_result>");
        append_tool_result_text(&b, m->content ? m->content : "");
        ember_buf_puts(&b, "</tool_result>");
    }
    ember_buf_puts(&b, ASST);
    ember_buf_puts(&b, enable_thinking ? "<think>" : "</think>");
    char *s = ember_buf_take(&b);
    return s ? s : strdup("");
}

char *ember_render_invalid_tool_recovery_suffix(
        const ember_chat_request *req, bool close_thinking,
        const char *detail) {
    if (!req || req->n_messages < 0 ||
        (req->n_messages > 0 && !req->messages)) return NULL;
    // The DeepSeek recovery turn is DSML-specific. Qwen malformed-call retry
    // remains disabled until its executable parser can validate the complete
    // XML block; returning NULL makes the existing caller stop the retry.
    if (req->prompt_profile == EMBER_PROMPT_QWEN4_CHATML) return NULL;
    ember_buf error = {0};
    ember_buf_puts(&error, "Tool error: invalid DSML tool call");
    if (detail && detail[0]) {
        ember_buf_puts(&error, ": ");
        ember_buf_puts(&error, detail);
    }
    ember_buf_puts(&error,
        "\nThe previous assistant output was not executed because its tool-call "
        "syntax was malformed. Emit a new valid DSML tool call, or answer "
        "normally if no tool is needed.");
    // ds4_server repeats its entire rendered system region here. agent gateway' system
    // and tool contract is thousands of tokens, so copying it made one malformed
    // call add ~6.2k prompt tokens and increased exact prompt-echo risk. The
    // original contract and schemas remain immediately earlier in context; keep
    // only the compact corrective turn, as ds4_agent.c:8681-8691 does.

    ember_buf suffix = {0};
    if (close_thinking) ember_buf_puts(&suffix, "</think>");
    ember_buf_puts(&suffix, EOS USER "<tool_result>");
    append_tool_result_text(&suffix, error.ptr ? error.ptr : "");
    ember_buf_puts(&suffix, "</tool_result>" ASST);
    ember_buf_puts(&suffix,
                   req->thinking_enabled ? "<think>" : "</think>");
    ember_buf_free(&error);
    char *s = ember_buf_take(&suffix);
    return s ? s : strdup("");
}

bool ember_prompt_ends_in_open_think(const char *prompt) {
    if (!prompt) return false;
    size_t end = strlen(prompt);
    while (end > 0) {
        char c = prompt[end - 1];
        if (c != ' ' && c != '\n' && c != '\r' && c != '\t') break;
        end--;
    }
    const char *open = "<think>";
    size_t olen = 7;
    return end >= olen && memcmp(prompt + end - olen, open, olen) == 0;
}
