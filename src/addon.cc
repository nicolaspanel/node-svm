#ifndef _LIBSVM_ADDON_H
#define _LIBSVM_ADDON_H
#include "node-svm/node-svm.h"

extern "C" {
  void InitAll(Local<Object> exports) {
    NodeSvm::Init(exports);
  }
}

#ifdef NODE_MODULE
NODE_MODULE(addon, InitAll)
#endif

#endif /* _LIBSVM_ADDON_H */
