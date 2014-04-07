#ifndef _NODE_SVM_H
#define _NODE_SVM_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <node.h>
#include <assert.h>
#include "../../node_modules/nan/nan.h"
#include "../libsvm-318/svm.h"

using namespace v8;

class NodeSvm : public node::ObjectWrap
{
  public:
    static void Init(Handle<Object> exports);
    static NAN_METHOD(SetParameters);
    static NAN_METHOD(Train);
    static NAN_METHOD(TrainAsync);
    static NAN_METHOD(IsTrained);
    static NAN_METHOD(GetLabels);
    static NAN_METHOD(GetKernelType);
    static NAN_METHOD(GetSvmType);
    static NAN_METHOD(Predict);
    static NAN_METHOD(PredictAsync);
    static NAN_METHOD(PredictProbabilities);
    static NAN_METHOD(PredictProbabilitiesAsync);
    static NAN_METHOD(SaveToFile);
    static NAN_METHOD(LoadFromFile);
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
    int getClassNumber(){
      if(model==NULL)
        return 0;
      return model->nr_class;
    }
    double getLabel(int index){
      if(model==NULL)
        return 0.0;
      return model->label[index];
    };
    int saveModel(const char *fileName){
      return svm_save_model(fileName, model);
    };
    void loadModel(const char *fileName){
      model = svm_load_model(fileName);
      assert(model!=NULL);  
      params = &model->param;
      assert(params!=NULL);  
    };
    Handle<Value> setSvmProblem(Local<Array> dataset){
      NanScope();
      struct svm_problem *prob = new svm_problem();
      prob->l = 0;

      if (dataset->Length() == 0)
        return ThrowException(Exception::Error(String::New("Training data set is empty")));      

      // check data structure and assign Y
      prob->l= dataset->Length();
      prob->y = new double[prob->l];
      int nb_features = -1;
      for (unsigned i=0; i < dataset->Length(); i++) {
        Local<Value> t = dataset->Get(i);
        if (!t->IsArray()) {
          return ThrowException(Exception::Error(String::New("First argument should be 2d-array (training data set)")));
        }
        Local<Array> ex = Array::Cast(*t);
        if (ex->Length() != 2) {
          return ThrowException(Exception::Error(String::New("Incorrect dataset (should be composed of two columns)")));
        }

        Local<Value> tin = ex->Get(0);
        Local<Value> tout = ex->Get(1);
        if (!tin->IsArray() || !tout->IsNumber()) {
          return ThrowException(Exception::Error(String::New("Incorrect dataset (first column should be an array and second column should be a number)")));
        }
        Local<Array> x = Array::Cast(*tin);
        if (nb_features == -1){
          nb_features = x->Length();
        }
        else{
          if (nb_features != (int)x->Length()){
            return ThrowException(Exception::Error(String::New("Incorrect dataset (all Xies should have the same length)")));
          }
        }
      }

      // Asign X and Y
      prob->x = new svm_node*[dataset->Length()];
      for(unsigned i = 0; i < dataset->Length(); i++) {
        prob->x[i] = new svm_node[nb_features + 1];
        Local<Array> ex = Array::Cast(*dataset->Get(i));
        Local<Array> x = Array::Cast(*ex->Get(0));
        for(unsigned j = 0; j < x->Length(); ++j) {
          double xi = x->Get(j)->NumberValue();
          prob->x[i][j].index = j+1;
          prob->x[i][j].value = xi;
        }
        prob->x[i][x->Length()].index = -1;

        double y = ex->Get(1)->NumberValue();
        prob->y[i] = y;
      }
      trainingProblem = prob;
      NanReturnUndefined();
    };

    void train(){
      model = svm_train(trainingProblem, params);
    };
    
    double predict(svm_node *x){
      return svm_predict(model, x);
    }
    void predictProbabilities(svm_node *x, double* prob_estimates){
      svm_predict_probability(model,x,prob_estimates);
    };
    void getSvmNodes(Local<Array> inputs, svm_node *nodes){      
      for (unsigned j=0; j < inputs->Length(); j++){
        double xi = inputs->Get(j)->NumberValue();
        nodes[j].index = j+1;
        nodes[j].value = xi;
      }
      nodes[inputs->Length()].index = -1;
    };
  private:
    ~NodeSvm();
    struct svm_parameter *params;
    struct svm_model *model;
    struct svm_problem *trainingProblem;
    static Persistent<Function> constructor;

};

#endif /* _NODE_SVM_H */