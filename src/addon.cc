#include "common.h"
#include "node-svm/node-svm.h"

extern "C" {
  void InitAll(Handle<Object> exports) {
    v8::HandleScope scope;
    NodeSvm::Init(exports);
  }
}

#ifdef NODE_MODULE
NODE_MODULE(addon, InitAll)
#endif
