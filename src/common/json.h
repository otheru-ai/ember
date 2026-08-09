// Minimal recursive-descent JSON parser → tagged-union DOM. Enough to parse
// OpenAI/Anthropic chat-completion request bodies (objects, arrays, strings
// with escapes incl. \uXXXX, numbers, bool, null). Hand-written, no deps, like
// ds4's parser. Values borrow into a single arena freed by ember_json_free.
#ifndef EMBER_JSON_H
#define EMBER_JSON_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    EMBER_JSON_NULL,
    EMBER_JSON_BOOL,
    EMBER_JSON_NUMBER,
    EMBER_JSON_STRING,
    EMBER_JSON_ARRAY,
    EMBER_JSON_OBJECT
} ember_json_type;

typedef struct ember_json ember_json;

struct ember_json {
    ember_json_type type;
    union {
        bool        b;
        double      num;
        char       *str;      // NUL-terminated, unescaped (OBJECT keys use `str` on members)
        struct {              // ARRAY / OBJECT
            ember_json **items;   // OBJECT members carry a `key`
            char       **keys;    // OBJECT only (parallel to items); NULL for ARRAY
            int          count;
        } arr;
    } u;
    char *num_raw;  // NUMBER: original token text (owned) so dump is byte-exact
};

// Parse `text`. Returns NULL on syntax error. Free the result with
// ember_json_free (frees the whole tree).
ember_json *ember_json_parse(const char *text);
void        ember_json_free(ember_json *v);

// Object member lookup (NULL if not an object or key absent).
const ember_json *ember_json_get(const ember_json *obj, const char *key);

// Typed accessors with defaults (return `dflt` on type mismatch / NULL).
const char *ember_json_str(const ember_json *v, const char *dflt);
double      ember_json_num(const ember_json *v, double dflt);
bool        ember_json_bool(const ember_json *v, bool dflt);
int         ember_json_len(const ember_json *v);            // array/object count
const ember_json *ember_json_at(const ember_json *v, int i);  // array element
const char *ember_json_key(const ember_json *obj, int i);   // object member key by index

// True when any object in `v` contains the same member name more than once.
// The general request parser remains RFC-8259 tolerant, but executable tool
// arguments use this to reject first-key/last-key parser differentials.
bool ember_json_has_duplicate_keys(const ember_json *v);

// Serialize `v` back to a compact JSON string (malloc'd; caller frees).
char *ember_json_dump(const ember_json *v);

#endif  // EMBER_JSON_H
