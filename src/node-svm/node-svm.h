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
    static NAN_METHOD(GetLabels);
    static NAN_METHOD(GetKernelType);
    static NAN_METHOD(GetSvmType);
    static NAN_METHOD(GetAccuracy);
    static NAN_METHOD(Predict);
    static NAN_METHOD(PredictAsync);
    static NAN_METHOD(PredictProbabilities);
    static NAN_METHOD(New);

    bool isTrained(){return model != NULL;};
    bool hasParameters(){return params != NULL;};
    bool isClassificationSVM(){
        if(!hasParameters())
            return false;
        int svm_type = params->svm_type;
        if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
            return false;
        return true;
    };
    bool isRegressionSVM(){ return !isClassificationSVM();};
    int getSvmType(){
        return params->svm_type;
    };
    int getKernelType(){
        return params->kernel_type;
    };
    struct svm_parameter *params;
    struct svm_model *model;
  private:
    ~NodeSvm();
    static Persistent<Function> constructor;
};

#endif /* _NODE_SVM_H */