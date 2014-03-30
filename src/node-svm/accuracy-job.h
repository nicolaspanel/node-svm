#ifndef _ACCURACY_JOB_H
#define _ACCURACY_JOB_H

#include "../common.h"
#include "node-svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class AccuracyJob : public NanAsyncWorker {
public:
  AccuracyJob(Local<Array> testset, NodeSvm *obj, NanCallback *callback) : NanAsyncWorker(callback){
    nb_examples = testset->Length();
    test_set = new svm_problem();
    test_set->l = nb_examples;
    test_set->y = Malloc(double,nb_examples);
    test_set->x = Malloc(struct svm_node *,nb_examples);
    
    for (unsigned i=0; i < nb_examples; i++) {
      Local<Object> t = testset->Get(i)->ToObject();
      test_set->y[i] = t->Get(String::New("y"))->NumberValue();
      Local<Array> x = Array::Cast(*t->Get(String::New("x"))->ToObject());
      test_set->x[i] = Malloc(struct svm_node,x->Length());
      for (unsigned j=0; j < x->Length(); j++){
        test_set->x[i][j].index = j+1;
        test_set->x[i][j].value = x->Get(j)->NumberValue();
      }
    }
    model = obj->model;
  }
  ~AccuracyJob() {}

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
    int correct = 0;
    int total = 0;
    int j;
    
    for (unsigned i=0; i < nb_examples; i++) {
      double target_label, predict_label;
      predict_label = svm_predict(model,test_set->x[i]);

      if(predict_label == test_set->y[i])
        ++correct;
      ++total;
    }
    accuracy = (double)correct/total;
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
   unsigned nb_examples;
   svm_model *model;
   svm_problem *test_set;
   double accuracy;
 };

#endif /* _ACCURACY_JOB_H */