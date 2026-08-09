// Compile a request's tool schemas into a DSML grammar for constrained decoding.
//
// tool_schema.c is fail-CLOSED: it inspects arguments after generation and
// rejects what does not conform. By then the tokens are spent, and production
// shows the model losing a whole turn to a block it was never able to complete
// -- most often a structurally perfect <invoke> with a required <parameter>
// simply absent. This header is the fail-OPEN counterpart: a grammar that makes
// those shapes unrepresentable, so the sampler cannot emit them in the first
// place. The two are complementary, not redundant; the validator stays as the
// trust boundary because a grammar constrains structure, not semantics.
//
// The emitted grammar closes exactly the observed failure modes:
//
//   <invoke name="terminal"></invoke>   cannot parse -- `command` is required,
//                                       so the closing tag is not reachable
//                                       until a <parameter> has been emitted
//   <invoke ...><invoke ...>            cannot parse -- invoke does not nest
//
// ORDERING is the one place this is deliberately tighter than JSON Schema.
// Schema requires properties to be PRESENT but leaves their order free, and
// "all of these in any order" needs a permutation expansion that is factorial
// in the property count. So this emits a canonical order instead: required
// properties in schema order, then optional ones, each skippable. Every string
// the grammar accepts is schema-valid; the cost is refusing an ordering the
// model might have preferred. With most tools carrying one or two required
// properties that is a good trade -- revisit if a tool appears where order
// genuinely varies.
//
// A property whose schema declares no type is left unconstrained between the
// string and JSON forms (both string="true" and string="false" are accepted),
// because guessing one would reject valid output.
//
// CONDITIONAL REQUIREMENTS. A tool may only require a property once a
// discriminator takes a particular value:
//
//   "required": ["mode"],
//   "allOf": [{"if":   {"properties": {"mode": {"const": "replace"}}},
//              "then": {"required": ["path", "old_string", "new_string"]}}]
//
// A flat `required` cannot express that, and production showed the cost: the
// model emitted {mode, new_string} seventeen times -- schema-valid, and refused
// by the tool with "path required" every time -- until a repeat guardrail
// stopped it. tool_schema.c already validates these correctly; without the
// same understanding here the grammar would happily generate the shape the
// validator then rejects.
//
// So a discriminated tool becomes an ALTERNATION, one branch per const value,
// each pinning the discriminator to its literal and requiring that branch's
// properties. Choosing the branch is what commits the model to its obligations,
// and a call that omits them has no derivation at all.
//
// Only that exact shape is recognised -- an `if` pinning exactly one property
// to a const, with every branch keyed off the same property. Anything else
// leaves the tool with its unconditional grammar: a grammar that guessed at a
// constraint it did not understand would reject valid calls, which is far worse
// than failing to constrain.
#ifndef EMBER_TOOL_GRAMMAR_H
#define EMBER_TOOL_GRAMMAR_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Build an EBNF grammar covering the tool_calls block for `tools_json` (the
// request's re-serialized tools array, as carried on ember_chat_request).
//
// `allow_parallel` mirrors the request's parallel_tool_calls: when false the
// grammar admits exactly one <invoke> per block.
//
// Returns a heap string the caller frees, or NULL when `tools_json` contains no
// tool with a usable schema -- in which case the caller must decode
// unconstrained rather than treat it as an error.
char *ember_tool_grammar_build(const char *tools_json, bool allow_parallel);

#ifdef __cplusplus
}
#endif

#endif
