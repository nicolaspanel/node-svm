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
    static NAN_METHOD(GetModel);
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

    Local<Object> getModel(){
      Local<Object> obj = NanNew<Object>();
      obj->Set(NanNew<String>("nrClass"), NanNew<Number>(model->nr_class));
      obj->Set(NanNew<String>("l"), NanNew<Number>(model->l));

      // Create a new array for support vectors
      Handle<Array> supportVectors = Array::New(model->l);
      const double * const *sv_coef = model->sv_coef;
      const svm_node * const *SV = model->SV;
      const int nb_outputs = model->nr_class - 1;
      for(int i=0;i<model->l;i++)
      {
        Handle<Array> outputs = Array::New(nb_outputs);
        for(int j=0; j < nb_outputs ; j++)
          outputs->Set(j, NanNew<Number>(sv_coef[j][i]));

        const svm_node *p = SV[i];

//        if(param.kernel_type == PRECOMPUTED)
//          fprintf(fp,"0:%d ",(int)(p->value));
//        else
        int max_index = 0;
        int nb_index = 0;
        while(p->index != -1)
        {
          nb_index++;
          if (p->index > max_index){
            max_index = p->index;
          }
          p++;
        }
//        fprintf(fp, "\n");
        Handle<Array> inputs = Array::New(nb_index);
        int p_i = 0;
        for (int k=0; k < max_index ; k++)
        {
          if (k+1 == model->SV[i][p_i].index){
            inputs->Set(k, NanNew<Number>(SV[i][p_i].value));
            p_i++;
          }
          else {
            inputs->Set(k, NanNew<Number>(0));
          }
        }
        Handle<Array> example = Array::New(2);
        example->Set(0, inputs);
        example->Set(1, outputs);
        supportVectors->Set(i, example);
      }
      obj->Set(NanNew<String>("supportVectors"), supportVectors);

      if(model->nSV)
      {
        Handle<Array> nbSupportVectors = Array::New(model->nr_class);
        for(int i=0; i < model->nr_class ; i++)
        {
          nbSupportVectors->Set(i, NanNew<Number>(model->nSV[i]));
        }
        obj->Set(NanNew<String>("nbSupportVectors"), nbSupportVectors);
      }
      if(model->label)
      {
        Handle<Array> labels = Array::New(model->nr_class);
        for(int i=0 ; i < model->nr_class ; i++)
          labels->Set(i, NanNew<Number>(model->label[i]));
        obj->Set(NanNew<String>("labels"), labels);
      }

      if(model->probA) // regression has probA only
      {
        int n = model->nr_class*(model->nr_class-1)/2;
        Handle<Array> probA = Array::New(n);
        for(int i=0 ; i < n ; i++)
          probA->Set(i, NanNew<Number>(model->probA[i]));
        obj->Set(NanNew<String>("probA"), probA);
      }
      else
      {
        obj->Set(NanNew<String>("probA"), NanUndefined());
      }

      if(model->probB)
      {
        int n = model->nr_class*(model->nr_class-1)/2;
        Handle<Array> probB = Array::New(n);
        for(int i=0 ; i < n ; i++)
          probB->Set(i, NanNew<Number>(model->probB[i]));
        obj->Set(NanNew<String>("probB"), probB);
      }
      else
      {
        obj->Set(NanNew<String>("probB"), NanUndefined());
      }

      if(model->rho)
      {
        int n = model->nr_class*(model->nr_class-1)/2;
        Handle<Array> rho = Array::New(n);
        for(int i=0 ; i < n ; i++)
          rho->Set(i, NanNew<Number>(model->rho[i]));
        obj->Set(NanNew<String>("rho"), rho);
      }
      else
      {
        obj->Set(NanNew<String>("rho"), NanUndefined());
      }

      Local<Object> params = NanNew<Object>();
      params->Set(NanNew<String>("svmType"), NanNew<Number>(model->param.svm_type));
      params->Set(NanNew<String>("kernelType"), NanNew<Number>(model->param.kernel_type));
      params->Set(NanNew<String>("degree"), NanNew<Number>(model->param.degree)); /* for poly */
      params->Set(NanNew<String>("gamma"), NanNew<Number>(model->param.gamma));
      params->Set(NanNew<String>("r"), NanNew<Number>(model->param.coef0));
      params->Set(NanNew<String>("C"), NanNew<Number>(model->param.C));
      params->Set(NanNew<String>("nu"), NanNew<Number>(model->param.nu));
      params->Set(NanNew<String>("p"), NanNew<Number>(model->param.p));

      Handle<Array> weightLabels = Array::New(model->param.nr_weight);
      Handle<Array> weights = Array::New(model->param.nr_weight);
      for(int i=0 ; i < model->param.nr_weight ; i++){
        weightLabels->Set(i, NanNew<Number>(model->param.weight_label[i]));
        weights->Set(i, NanNew<Number>(model->param.weight[i]));
      }
      params->Set(NanNew<String>("weightLabels"), weightLabels);
      params->Set(NanNew<String>("weighs"), weights);

      params->Set(NanNew<String>("cacheSize"), NanNew<Number>(model->param.cache_size));
      params->Set(NanNew<String>("eps"), NanNew<Number>(model->param.eps));
      if (model->param.shrinking){
        params->Set(NanNew<String>("shrinking"), NanTrue());
      }
      else {
        params->Set(NanNew<String>("shrinking"), NanFalse());
      }

      if (model->param.probability){
        params->Set(NanNew<String>("probability"), NanTrue());
      }
      else {
        params->Set(NanNew<String>("probability"), NanFalse());
      }

      obj->Set(NanNew<String>("params"), params);
      return obj;
    };
  private:
    ~NodeSvm();
    struct svm_parameter *params;
    struct svm_model *model;
    struct svm_problem *trainingProblem;
    static Persistent<Function> constructor;

};

#endif /* _NODE_SVM_H */