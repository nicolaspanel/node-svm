#ifndef _NODE_SVM_H
#define _NODE_SVM_H

#include "../common.h"

using namespace v8;

class NodeSvm : public node::ObjectWrap
{
  public:
    static void Init(Handle<Object> exports);
    static NAN_METHOD(SetParameters);
    static NAN_METHOD(Train);
    static NAN_METHOD(Predict);
    static NAN_METHOD(PredictAsync);
    static NAN_METHOD(New);
    
    void trainInstance(svm_problem *problem);
    double predict();
    
    svm_parameter *params;
    svm_model *model;
  private:
    ~NodeSvm();
    static Persistent<Function> constructor;
};

#endif /* _NODE_SVM_H */