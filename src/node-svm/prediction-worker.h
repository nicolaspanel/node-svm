#ifndef _NODE_SVM_PREDICTION_WORKER_H
#define _NODE_SVM_PREDICTION_WORKER_H

#include "node-svm.h"

using namespace v8;

class PredictionWorker : public NanAsyncWorker {
 public:
  PredictionWorker(NodeSvm *svm, Local<Array> inputs, NanCallback *callback)
    : NanAsyncWorker(callback) {
      obj = svm;
      x = new svm_node[inputs->Length() + 1];
      obj->getSvmNodes(inputs, x);
    }
  ~PredictionWorker() {
    delete[] x;
  }

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
    prediction = obj->predict(x);
  }

  // Executed when the async work is complete
  // this function will be run inside the main event loop
  // so it is safe to use V8 again
  void HandleOKCallback () {
    NanScope();
    Local<Value> argv[] = {
      Number::New(prediction)
    };
    callback->Call(1, argv);
  };

 private:
  NodeSvm *obj;
  svm_node *x;
  double prediction;
};

#endif /* _NODE_SVM_PREDICTION_WORKER_H */