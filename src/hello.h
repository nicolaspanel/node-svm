#include <node.h>
#include "../node_modules/nan/nan.h"

using namespace v8;

// Simple synchronous access to the `Estimate()` function
NAN_METHOD(Hello) {
  NanScope();
  NanReturnValue(String::New("World!"));
}