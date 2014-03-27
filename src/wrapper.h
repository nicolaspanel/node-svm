#ifndef _WRAPPER_H
#define _WRAPPER_H

#include <node.h>
#include "../node_modules/nan/nan.h"
#include "./libsvm-317/svm.h"

using namespace v8;

class TrainWorker : public NanAsyncWorker {
 public:
  TrainWorker(svm_problem *prob, svm_parameter *params, NanCallback *callback)
    : NanAsyncWorker(callback){
      _problem = prob;
      _params = params;
  }
  ~TrainWorker() {}

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
    //_model = svm_train(_problem, _params);
  }

  // Executed when the async work is complete
  // this function will be run inside the main event loop
  // so it is safe to use V8 again
  void HandleOKCallback () {
    NanScope();
    Local<Value> argv[] = {
    };
    callback->Call(0, argv);
  };
 private:
  svm_problem *_problem;
  svm_parameter *_params;
  svm_model *_model;
};


// Asynchronous access to the `Estimate()` function

// struct svm_parameter
// {
//   int svm_type;
//   int kernel_type;
//   int degree; /* for poly */
//   double gamma; /* for poly/rbf/sigmoid */
//   double coef0; /* for poly/sigmoid */

//   /* these are for training only */
//   double cache_size; /* in MB */
//   double eps; /* stopping criteria */
//   double C;  for C_SVC, EPSILON_SVR and NU_SVR 
//   int nr_weight;    /* for C_SVC */
//   int *weight_label;  /* for C_SVC */
//   double* weight;   /* for C_SVC */
//   double nu;  /* for NU_SVC, ONE_CLASS, and NU_SVR */
//   double p; /* for EPSILON_SVR */
//   int shrinking;  /* use the shrinking heuristics */
//   int probability; /* do probability estimates */
// };
NAN_METHOD(svmTrain) {
  NanScope();

  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  svm_problem *problem = new svm_problem();

  Local<Object> parameters = *args[1]->ToObject();
  svm_parameter *params = new svm_parameter();
  
  if (parameters->Has(String::New("type"))) {
    params->svm_type = parameters->Get(String::New("type"))->IntegerValue();
  }
  if (parameters->Has(String::New("kernel"))) {
    params->kernel_type = parameters->Get(String::New("kernel"))->IntegerValue();
  }

  NanCallback *callback = new NanCallback(args[2].As<Function>());
  
  NanAsyncQueueWorker(new TrainWorker(problem, params, callback));
  NanReturnUndefined();
}

#endif /* _WRAPPER_H */