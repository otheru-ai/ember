#include <stdio.h>
#include <string.h>
#include "../src/server/http.h"
static int g_pass=0,g_fail=0;
#define CHECK(c,m) do{ if(c)g_pass++; else{g_fail++;printf("  FAIL: %s\n",m);} }while(0)
int main(void){
    printf("ember http tests\n");
    char raw[] = "POST /v1/chat/completions HTTP/1.1\r\n"
                 "Host: localhost\r\nContent-Type: application/json\r\n"
                 "Content-Length: 7\r\n\r\n{\"a\":1}";
    ember_http_request req;
    size_t off = ember_http_parse(raw, strlen(raw), &req);
    CHECK(off>0, "parse ok");
    CHECK(strcmp(req.method,"POST")==0, "method");
    CHECK(strcmp(req.path,"/v1/chat/completions")==0, "path");
    CHECK(strcmp(ember_http_header(&req,"content-type"),"application/json")==0, "hdr case-insensitive");
    CHECK(req.body_len==7 && strncmp(req.body,"{\"a\":1}",7)==0, "body + content-length");
    printf("──────────────────────────────\n  %d passed, %d failed\n",g_pass,g_fail);
    return g_fail?1:0;
}
