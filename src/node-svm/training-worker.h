#ifndef _NODE_SVM_TRAINING_WORKER_H
#define _NODE_SVM_TRAINING_WORKER_H

#include "node-svm.h"

using namespace v8;

class TrainingWorker : public Nan::AsyncWorker {
 public:
  TrainingWorker(NodeSvm *svm, Local<Array> dataset, Nan::Callback *callback)
    : Nan::AsyncWorker(callback) {
      obj = svm;
      obj->setSvmProblem(dataset);
    }
  ~TrainingWorker() {}

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
    obj->train();
  }

  // Executed when the async work is complete
  // this function will be run inside the main event loop
  // so it is safe to use V8 again
  void HandleOKCallback () {
    Nan::HandleScope scope;

#ifdef _WIN32
    // On windows you get "error C2466: cannot allocate an array of constant size 0" and we use a pointer
    Local<Value>* argv;
#else
    Local<Value> argv[0];
#endif

    callback->Call(0, argv);
  };

 private:
  NodeSvm *obj;
};

#endif /* _NODE_SVM_TRAINING_WORKER_H */
