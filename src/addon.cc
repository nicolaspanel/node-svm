#include <node.h>
#include "../node_modules/nan/nan.h"
#include "hello.h"

using namespace v8;

void InitAll(Handle<Object> exports) {
  exports->Set(NanSymbol("hello"),
    FunctionTemplate::New(Hello)->GetFunction());
}

NODE_MODULE(addon, InitAll)