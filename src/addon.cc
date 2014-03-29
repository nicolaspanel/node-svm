#include "common.h"
#include "hello.h"
#include "helloAsync.h"
#include "node-svm/node-svm.h"

extern "C" {
  void InitAll(Handle<Object> exports) {
    v8::HandleScope scope;
    exports->Set(NanSymbol("hello"), FunctionTemplate::New(Hello)->GetFunction());
    exports->Set(NanSymbol("helloAsync"), FunctionTemplate::New(HelloAsync)->GetFunction());

    NodeSvm::Init(exports);
  }
}

#ifdef NODE_MODULE
NODE_MODULE(addon, InitAll)
#endif
