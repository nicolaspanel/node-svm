#ifndef _NODE_SVM_PREOBABILITY_PREDICTION_WORKER_H
#define _NODE_SVM_PREOBABILITY_PREDICTION_WORKER_H

#include "node-svm.h"

using namespace v8;

class ProbabilityPredictionWorker : public Nan::AsyncWorker {
 public:
  ProbabilityPredictionWorker(NodeSvm *svm, Local<Array> inputs, Nan::Callback *callback)
    : Nan::AsyncWorker(callback) {
      obj = svm;

      x = new svm_node[inputs->Length() + 1];
      obj->getSvmNodes(inputs, x);

      nbClass = obj->getClassNumber();
      prob_estimates = new double[nbClass];
    }
  ~ProbabilityPredictionWorker() {
    delete[] x;
    delete[] prob_estimates;
  }

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
    obj->predictProbabilities(x, prob_estimates);
  }

  // Executed when the async work is complete
  // this function will be run inside the main event loop
  // so it is safe to use V8 again
  void HandleOKCallback () {
    Nan::HandleScope scope;
    // Create the result array
    Local<Array> probs = Nan::New<Array>(nbClass);
    for (int j=0; j < nbClass; j++){
      probs->Set(j, Nan::New<Number>(prob_estimates[j]));
    }
    Local<Value> argv[] = {
      probs
    };
    callback->Call(1, argv);
  };

 private:
  NodeSvm *obj;
  svm_node *x;
  double *prob_estimates;
  int nbClass;
};

#endif /* _NODE_SVM_PREOBABILITY_PREDICTION_WORKER_H */
