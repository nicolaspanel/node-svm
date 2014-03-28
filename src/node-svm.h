#ifndef _NODE_SVM_H
#define _NODE_SVM_H

#include <stdio.h>
#include <stdlib.h>
#include <node.h>
#include "../node_modules/nan/nan.h"
#include "./libsvm-317/svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

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
    svm_model *_model = svm_train(_problem, _params);
    save_status = svm_save_model("model.model", _model);
  }

  // Executed when the async work is complete
  // this function will be run inside the main event loop
  // so it is safe to use V8 again
  void HandleOKCallback () {
    NanScope();
    Local<Value> argv[] = {
       save_status == -1 ? String::New("failed to save the model") : String::New("")
    };
    callback->Call(1, argv);
  };
 private:
  svm_problem *_problem;
  svm_parameter *_params;
  int save_status;
};


NAN_METHOD(svmTrain) {
  NanScope();

  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  svm_problem *problem = new svm_problem();
  problem->l = dataset->Length();
  problem->y = Malloc(double,problem->l);
  problem->x = Malloc(struct svm_node *,problem->l);
  
  for (unsigned i=0; i < dataset->Length(); i++) {
    Local<Object> t = dataset->Get(i)->ToObject();
    problem->y[i] = t->Get(String::New("y"))->NumberValue();
    Local<Array> x = Array::Cast(*t->Get(String::New("x"))->ToObject());
    problem->x[i] = Malloc(struct svm_node,x->Length());
    for (unsigned j=0; j < x->Length(); j++){
      problem->x[i][j].index = j;
      problem->x[i][j].value = x->Get(j)->NumberValue();
    }
  }
  
  Local<Object> parameters = *args[1]->ToObject();
  svm_parameter *svm_params = new svm_parameter();
  
  svm_params->svm_type = parameters->Get(String::New("type"))->IntegerValue();
  svm_params->kernel_type = parameters->Get(String::New("kernel"))->IntegerValue();
  svm_params->degree = parameters->Get(String::New("degree"))->IntegerValue();
  svm_params->gamma = parameters->Get(String::New("gamma"))->NumberValue();
  svm_params->coef0 = parameters->Get(String::New("r"))->NumberValue();
  
  svm_params->cache_size = parameters->Get(String::New("cacheSize"))->NumberValue();
  svm_params->eps = parameters->Get(String::New("eps"))->NumberValue();
  svm_params->C = parameters->Get(String::New("C"))->NumberValue();
  svm_params->nu = parameters->Get(String::New("nu"))->NumberValue();
  svm_params->p = parameters->Get(String::New("p"))->NumberValue();
  svm_params->shrinking = parameters->Get(String::New("shrinking"))->IntegerValue();
  svm_params->probability = parameters->Get(String::New("probability"))->IntegerValue();

  NanCallback *callback = new NanCallback(args[2].As<Function>());
  
  NanAsyncQueueWorker(new TrainWorker(problem, svm_params, callback));
  NanReturnUndefined();
}

#endif /* _NODE_SVM_H */