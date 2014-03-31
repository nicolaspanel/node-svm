#ifndef _ACCURACY_JOB_H
#define _ACCURACY_JOB_H

#include "node-svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class AccuracyJob : public NanAsyncWorker {
public:
  AccuracyJob(svm_problem *prob , svm_model *model, NanCallback *callback) : NanAsyncWorker(callback){
    test_set = prob;
    _model = model;
  }
  ~AccuracyJob() {}

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
    int correct = 0;
    int total = 0;
    
    for (unsigned i=0; i < test_set->l; i++) {
      double predict_label = svm_predict(_model,test_set->x[i]);

      if(predict_label == test_set->y[i])
        ++correct;
      ++total;
    }
    accuracy = (double)correct/(double)total;
  }

   // Executed when the async work is complete
   // this function will be run inside the main event loop
   // so it is safe to use V8 again
   void HandleOKCallback () {
     NanScope();
     Local<Value> argv[] = {
        Number::New(accuracy)
     };
     callback->Call(1, argv);
   };
  private:
   svm_model *_model;
   svm_problem *test_set;
   double accuracy;
 };

#endif /* _ACCURACY_JOB_H */