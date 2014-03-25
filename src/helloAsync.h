#include <node.h>
#include "../node_modules/nan/nan.h"

using namespace v8;

class HelloWorker : public NanAsyncWorker {
 public:
  HelloWorker(NanCallback *callback)
    : NanAsyncWorker(callback){}
  ~HelloWorker() {}

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
    
  }

  // Executed when the async work is complete
  // this function will be run inside the main event loop
  // so it is safe to use V8 again
  void HandleOKCallback () {
    NanScope();
    Local<Value> argv[] = {
        NanNewLocal<Value>(Null()),
        String::New("World!")
    };
    callback->Call(1, argv);
  };
};


// Asynchronous access to the `Estimate()` function
NAN_METHOD(HelloAsync) {
  NanScope();

  NanCallback *callback = new NanCallback(args[0].As<Function>());

  NanAsyncQueueWorker(new HelloWorker(callback));
  NanReturnUndefined();
}