#ifndef EMBER_API_ADAPTERS_H
#define EMBER_API_ADAPTERS_H

#include <stdbool.h>

#include "chat_api.h"
#include "sse.h"

// Parse protocol-native bodies into Ember's protocol-neutral chat request.
// `err` receives a short client-facing reason on failure.
bool ember_responses_request_parse(const ember_json *root,
                                   ember_chat_request *out,
                                   char *err, size_t err_cap);
bool ember_anthropic_request_parse(const ember_json *root,
                                   ember_chat_request *out,
                                   char *err, size_t err_cap);
bool ember_completion_request_parse(const ember_json *root,
                                    ember_chat_request *out,
                                    char *err, size_t err_cap);

typedef struct {
    const char       *id;
    const char       *model;
    long              created;
    const char       *content;
    const char       *reasoning;
    ember_tool_calls *calls;
    const char       *finish_reason;
    const char       *stop_sequence;  // exact matched client stop, or NULL
    const char       *termination_reason;
    bool              reasoning_budget_exhausted;
    int               tool_loop_rounds;
    // false when the count came from the weaker call-signature signal.
    bool              tool_loop_identical;
    const char       *tool_loop_tool;
    int               prompt_tokens;
    int               completion_tokens;
    int               cached_tokens;
} ember_protocol_result;

typedef enum {
    EMBER_PROTOCOL_STREAM_NONE,
    EMBER_PROTOCOL_STREAM_REASONING,
    EMBER_PROTOCOL_STREAM_TEXT,
    EMBER_PROTOCOL_STREAM_TOOL,
} ember_protocol_stream_kind;

// State for a token-live native protocol stream. The shared SSE parser calls
// this through ember_sse_sink, so reasoning, visible text, and DSML tool
// arguments retain the same UTF-8/marker holdback guarantees as Chat.
typedef struct {
    const ember_chat_request *req;  // borrowed
    const char *id;                 // borrowed
    const char *model;              // borrowed
    long created;
    int prompt_tokens;
    int sequence;
    int output_index;               // Responses output[] index
    int block_index;                // Anthropic content[] index
    char reasoning_id[72];
    char message_id[72];
    ember_protocol_stream_kind active;
    int active_tool_index;
    char *active_id;              // protocol item id (Responses fc_N)
    char *active_call_id;         // model-visible call id
    char *active_name;
    ember_buf active_value;
} ember_protocol_stream;

void ember_protocol_stream_init(ember_protocol_stream *st,
                                const ember_chat_request *req,
                                const char *id, const char *model,
                                long created, int prompt_tokens);
void ember_protocol_stream_free(ember_protocol_stream *st);
void ember_protocol_stream_bind(ember_protocol_stream *st,
                                ember_sse_stream *splitter);
void ember_protocol_stream_begin(ember_protocol_stream *st, ember_buf *out);
void ember_protocol_stream_finish(ember_protocol_stream *st,
                                  const ember_protocol_result *result,
                                  ember_buf *out);
void ember_protocol_stream_error(ember_protocol_stream *st,
                                 const char *code, const char *detail,
                                 bool retry_exhausted, ember_buf *out);

// Emit a complete native response. For a request that is already buffered,
// stream=true replays it through the same native lifecycle used by the live
// generation path.
bool ember_protocol_emit(int fd, const ember_chat_request *req,
                         const ember_protocol_result *result);

#endif
