#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../src/model/gguf.h"

static int g_pass=0,g_fail=0;
#define CHECK(c,m) do{ if(c)g_pass++; else{g_fail++;printf("  FAIL: %s\n",m);} }while(0)

static void malformed_header_tests(void) {
    char path[] = "/tmp/ember-gguf-test-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "create malformed GGUF fixture");
    if (fd < 0) return;
    FILE *f = fdopen(fd, "wb");
    uint32_t version = 3;
    uint64_t tensors = 0, kv = UINT64_MAX;
    fwrite("GGUF", 1, 4, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&tensors, sizeof(tensors), 1, f);
    fwrite(&kv, sizeof(kv), 1, f);
    fclose(f);
    gguf_file *g = ember_gguf_open(path);
    CHECK(g == NULL, "unbounded metadata count rejected");
    ember_gguf_free(g);
    unlink(path);

    strcpy(path, "/tmp/ember-gguf-test-XXXXXX");
    fd = mkstemp(path);
    CHECK(fd >= 0, "create truncated GGUF fixture");
    if (fd < 0) return;
    f = fdopen(fd, "wb");
    kv = 1;
    fwrite("GGUF", 1, 4, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&tensors, sizeof(tensors), 1, f);
    fwrite(&kv, sizeof(kv), 1, f);
    uint64_t impossible_string = 16;
    fwrite(&impossible_string, sizeof(impossible_string), 1, f);
    fclose(f);
    g = ember_gguf_open(path);
    CHECK(g == NULL, "truncated metadata string rejected");
    ember_gguf_free(g);
    unlink(path);
}

int main(int argc, char **argv) {
    printf("ember gguf tests\n");
    malformed_header_tests();
    const char *path = argc > 1 ? argv[1] :
        "/models/model.gguf";
    gguf_file *g = ember_gguf_open(path);
    if (!g) { printf("  SKIP: cannot open %s (run as root or pass a path)\n", path);
              return 0; }  // don't fail CI when the 95GB file isn't readable
    CHECK(g->version == 3, "gguf v3");
    CHECK(strcmp(ember_gguf_get_str(g,"general.architecture","")," deepseek4")!=0 &&
          strcmp(ember_gguf_get_str(g,"general.architecture",""),"deepseek4")==0,
          "architecture=deepseek4");
    CHECK(ember_gguf_get_int(g,"deepseek4.block_count",0)==43, "43 blocks");
    CHECK(ember_gguf_get_int(g,"deepseek4.vocab_size",0)==129280, "vocab 129280");
    CHECK(ember_gguf_get_int(g,"deepseek4.expert_count",0)==256, "256 experts");
    const gguf_kv *toks = ember_gguf_get(g,"tokenizer.ggml.tokens");
    CHECK(toks && toks->type==GGUF_ARRAY && toks->arr_type==GGUF_STRING &&
          toks->arr_len==129280, "tokens array materialized (129280)");
    CHECK(toks && toks->arr_str && toks->arr_str[0]!=NULL, "token[0] present");
    CHECK(g->n_tensors > 1000, "tensor directory read");
    // find a known tensor
    int found=0;
    for (uint64_t i=0;i<g->n_tensors;i++)
        if (strcmp(g->tensors[i].name,"blk.0.attn_output_b.weight")==0) found=1;
    CHECK(found, "known tensor blk.0.attn_output_b.weight located");
    printf("  arch=%s blocks=%ld vocab=%ld tensors=%lu data_off=%lu\n",
           ember_gguf_get_str(g,"general.architecture","?"),
           (long)ember_gguf_get_int(g,"deepseek4.block_count",-1),
           (long)ember_gguf_get_int(g,"deepseek4.vocab_size",-1),
           (unsigned long)g->n_tensors,(unsigned long)g->data_offset);
    ember_gguf_free(g);
    printf("──────────────────────────────\n  %d passed, %d failed\n",g_pass,g_fail);
    return g_fail?1:0;
}
