#ifndef _TRAINING_JOB_H
#define _TRAINING_JOB_H

#include "../common.h"
#include "node-svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class TrainingJob : public NanAsyncWorker {
public:
  TrainingJob(Local<Array> dataset, NodeSvm *obj, NanCallback *callback) : NanAsyncWorker(callback){
    problem = new svm_problem();
    problem->l = dataset->Length();
    problem->y = Malloc(double,problem->l);
    problem->x = Malloc(struct svm_node *,problem->l);
    
    for (unsigned i=0; i < dataset->Length(); i++) {
      Local<Object> t = dataset->Get(i)->ToObject();
      problem->y[i] = t->Get(String::New("y"))->NumberValue();
      Local<Array> x = Array::Cast(*t->Get(String::New("x"))->ToObject());
      problem->x[i] = Malloc(struct svm_node,x->Length());
      for (unsigned j=0; j < x->Length(); j++){
        problem->x[i][j].index = j+1;
        problem->x[i][j].value = x->Get(j)->NumberValue();
      }
    }
    _obj = obj;
  }
  ~TrainingJob() {}

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
  
    _obj->trainInstance(problem);
  }

   // Executed when the async work is complete
   // this function will be run inside the main event loop
   // so it is safe to use V8 again
   void HandleOKCallback () {
     NanScope();
     Local<Value> argv[] = {
        String::New("ko")
     };
     callback->Call(1, argv);
   };
  private:
   svm_problem * problem;
   NodeSvm *_obj;
 };

#endif /* _TRAINING_JOB_H */