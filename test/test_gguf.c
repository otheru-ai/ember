#include <stdio.h>
#include <string.h>
#include "../src/model/gguf.h"

static int g_pass=0,g_fail=0;
#define CHECK(c,m) do{ if(c)g_pass++; else{g_fail++;printf("  FAIL: %s\n",m);} }while(0)

int main(int argc, char **argv) {
    printf("ember gguf tests\n");
    const char *path = argc > 1 ? argv[1] :
        "/srv/lucebox/models/DeepSeek-V4-Flash-ROCMFP2-STRIX-abliterated.gguf";
    gguf_file *g = gguf_open(path);
    if (!g) { printf("  SKIP: cannot open %s (run as root or pass a path)\n", path);
              return 0; }  // don't fail CI when the 95GB file isn't readable
    CHECK(g->version == 3, "gguf v3");
    CHECK(strcmp(gguf_get_str(g,"general.architecture","")," deepseek4")!=0 &&
          strcmp(gguf_get_str(g,"general.architecture",""),"deepseek4")==0,
          "architecture=deepseek4");
    CHECK(gguf_get_int(g,"deepseek4.block_count",0)==43, "43 blocks");
    CHECK(gguf_get_int(g,"deepseek4.vocab_size",0)==129280, "vocab 129280");
    CHECK(gguf_get_int(g,"deepseek4.expert_count",0)==256, "256 experts");
    const gguf_kv *toks = gguf_get(g,"tokenizer.ggml.tokens");
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
           gguf_get_str(g,"general.architecture","?"),
           (long)gguf_get_int(g,"deepseek4.block_count",-1),
           (long)gguf_get_int(g,"deepseek4.vocab_size",-1),
           (unsigned long)g->n_tensors,(unsigned long)g->data_offset);
    gguf_free(g);
    printf("──────────────────────────────\n  %d passed, %d failed\n",g_pass,g_fail);
    return g_fail?1:0;
}
