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
    static NAN_METHOD(SetModel);
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
    void loadModelFromFile(const char *fileName){
      model = svm_load_model(fileName);
      assert(model!=NULL);
      params = &model->param;
      assert(params!=NULL);  
    };

    void setParameters(Local<Object> obj){
      struct svm_parameter *svm_params = new svm_parameter();
      svm_params->svm_type = C_SVC;
      svm_params->kernel_type = RBF;
      svm_params->degree = 3;
      svm_params->gamma = 0;  // 1/num_features
      svm_params->coef0 = 0;
      svm_params->nu = 0.5;
      svm_params->cache_size = 100;
      svm_params->C = 1;
      svm_params->eps = 1e-3;
      svm_params->p = 0.1;
      svm_params->shrinking = 1;
      svm_params->probability = 0;
      svm_params->nr_weight = 0;
      svm_params->weight_label = NULL;
      svm_params->weight = NULL;

      if (obj->Has(String::New("svmType"))){
        svm_params->svm_type = obj->Get(String::New("svmType"))->IntegerValue();
      }
      if (obj->Has(String::New("kernelType"))){
        svm_params->kernel_type = obj->Get(String::New("kernelType"))->IntegerValue();
      }
      if (obj->Has(String::New("degree"))){
        svm_params->degree = obj->Get(String::New("degree"))->IntegerValue();
      }
      if (obj->Has(String::New("gamma"))){
        svm_params->gamma = obj->Get(String::New("gamma"))->NumberValue();
      }
      if (obj->Has(String::New("r"))){
        svm_params->coef0 = obj->Get(String::New("r"))->NumberValue();
      }
      if (obj->Has(String::New("C"))){
        svm_params->C = obj->Get(String::New("C"))->NumberValue();
      }
      if (obj->Has(String::New("nu"))){
        svm_params->nu = obj->Get(String::New("nu"))->NumberValue();
      }
      if (obj->Has(String::New("epsilon"))){
        svm_params->p = obj->Get(String::New("epsilon"))->NumberValue();
      }
      if (obj->Has(String::New("cacheSize"))){
        svm_params->cache_size = obj->Get(String::New("cacheSize"))->NumberValue();
      }
      if (obj->Has(String::New("eps"))){
        svm_params->eps = obj->Get(String::New("eps"))->NumberValue();
      }
      if (obj->Has(String::New("shrinking"))){
        svm_params->shrinking = obj->Get(String::New("shrinking"))->IntegerValue();
      }
      if (obj->Has(String::New("probability"))){
        svm_params->probability = obj->Get(String::New("probability"))->IntegerValue();
      }

      const char *error_msg;
      error_msg =  svm_check_parameter(svm_params);
      std::cout << error_msg << std::endl;
      assert(!error_msg);
      params = svm_params;
    };

    void setModel(Local<Object> obj){
      assert(obj->Has(String::New("params")));
      assert(obj->Get(String::New("params"))->IsObject());

      setParameters(obj->Get(String::New("params"))->ToObject());
      assert(params!=NULL);

      struct svm_model *new_model = new svm_model();
      new_model->free_sv = 1;	// XXX
      new_model->rho = NULL;
      new_model->probA = NULL;
      new_model->probB = NULL;
      new_model->sv_indices = NULL;
      new_model->label = NULL;
      new_model->nSV = NULL;

      assert(obj->Has(String::New("l")));
      assert(obj->Get(String::New("l"))->IsInt32());
      new_model->l = obj->Get(String::New("l"))->IntegerValue();

      new_model->nr_class = obj->Get(String::New("nrClass"))->IntegerValue();
      int n = new_model->nr_class * (new_model->nr_class-1)/2;

      // rho
      assert(obj->Has(String::New("rho")));
      assert(obj->Get(String::New("rho"))->IsArray());
      Local<Array> rho = Array::Cast(*obj->Get(String::New("rho"))->ToObject());
      assert(rho->Length()==n);
      new_model->rho = new double[n];
      for(int i=0;i<n;i++){
        Local<Value> elt = rho->Get(i);
        assert(elt->IsNumber());
        new_model->rho[i] = elt->NumberValue();
      }

      // classes
      assert(obj->Has(String::New("labels")));
      assert(obj->Get(String::New("labels"))->IsArray());
      Local<Array> labels = Array::Cast(*obj->Get(String::New("labels"))->ToObject());
			assert(labels->Length()==new_model->nr_class);
			new_model->label = new int[new_model->nr_class];
			for(int i=0;i<new_model->nr_class;i++){
				Local<Value> elt = labels->Get(i);
        assert(elt->IsInt32());
				new_model->label[i] = elt->IntegerValue();
			}

			// probA
			if (obj->Has(String::New("probA"))){
			  assert(obj->Get(String::New("probA"))->IsArray());
        Local<Array> probA = Array::Cast(*obj->Get(String::New("probA"))->ToObject());
        assert(probA->Length()==n);
        new_model->probA = new double[n];
        for(int i=0;i<n;i++){
          Local<Value> elt = probA->Get(i);
          assert(elt->IsNumber());
          new_model->probA[i] = elt->NumberValue();
        }
			}

			// probB
      if (obj->Has(String::New("probB"))){
        assert(obj->Get(String::New("probB"))->IsArray());
        Local<Array> probB = Array::Cast(*obj->Get(String::New("probB"))->ToObject());
        assert(probB->Length()==n);
        new_model->probB = new double[n];
        for(int i=0;i<n;i++){
          Local<Value> elt = probB->Get(i);
          assert(elt->IsNumber());
          new_model->probB[i] = elt->NumberValue();
        }
      }

      // nSV
      assert(obj->Has(String::New("nbSupportVectors")));
      assert(obj->Get(String::New("nbSupportVectors"))->IsArray());
      Local<Array> nbSupportVectors = Array::Cast(*obj->Get(String::New("nbSupportVectors"))->ToObject());
      assert(nbSupportVectors->Length()==new_model->nr_class);
      new_model->nSV = new int[new_model->nr_class];
      for(int i=0;i<new_model->nr_class;i++){
        Local<Value> elt = nbSupportVectors->Get(i);
        assert(elt->IsInt32());
        new_model->nSV[i] = elt->IntegerValue();
      }

      // SV
      assert(obj->Has(String::New("supportVectors")));
      assert(obj->Get(String::New("supportVectors"))->IsArray());
      Local<Array> supportVectors = Array::Cast(*obj->Get(String::New("supportVectors"))->ToObject());
      assert(supportVectors->Length()==new_model->l);
      int m = new_model->nr_class - 1;
      int l = new_model->l;
      new_model->sv_coef = new double *[m];
      for(int i=0; i < m ;i++)
      		new_model->sv_coef[i] = new double[l];
      new_model->SV = new svm_node*[l];
      for(int i = 0; i < l; i++) {
        Local<Array> ex = Array::Cast(*supportVectors->Get(i));
        assert(ex->Length()==2);
        Local<Array> x = Array::Cast(*ex->Get(0));
        Local<Array> y = Array::Cast(*ex->Get(1));

        new_model->SV[i] = new svm_node[x->Length() + 1];
        for(unsigned j = 0; j < x->Length(); ++j) {
          new_model->SV[i][j].index = j+1;
          new_model->SV[i][j].value = x->Get(j)->NumberValue();
        }
        new_model->SV[i][x->Length()].index = -1;

        for(int j=0; j < m ;j++)
          new_model->sv_coef[j][i] = y->Get(j)->NumberValue();
      }


      model = new_model;
      model->param = *params;
      assert(model!=NULL);
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

      if(model->probB)
      {
        int n = model->nr_class*(model->nr_class-1)/2;
        Handle<Array> probB = Array::New(n);
        for(int i=0 ; i < n ; i++)
          probB->Set(i, NanNew<Number>(model->probB[i]));
        obj->Set(NanNew<String>("probB"), probB);
      }

      if(model->rho)
      {
        int n = model->nr_class*(model->nr_class-1)/2;
        Handle<Array> rho = Array::New(n);
        for(int i=0 ; i < n ; i++)
          rho->Set(i, NanNew<Number>(model->rho[i]));
        obj->Set(NanNew<String>("rho"), rho);
      }

      Local<Object> parameters = NanNew<Object>();
      parameters->Set(NanNew<String>("svmType"), NanNew<Number>(model->param.svm_type));
      parameters->Set(NanNew<String>("kernelType"), NanNew<Number>(model->param.kernel_type));
      parameters->Set(NanNew<String>("degree"), NanNew<Number>(model->param.degree)); /* for poly */
      parameters->Set(NanNew<String>("gamma"), NanNew<Number>(model->param.gamma));
      parameters->Set(NanNew<String>("r"), NanNew<Number>(model->param.coef0));
      parameters->Set(NanNew<String>("C"), NanNew<Number>(model->param.C));
      parameters->Set(NanNew<String>("nu"), NanNew<Number>(model->param.nu));
      parameters->Set(NanNew<String>("p"), NanNew<Number>(model->param.p));

      Handle<Array> weightLabels = Array::New(model->param.nr_weight);
      Handle<Array> weights = Array::New(model->param.nr_weight);
      for(int i=0 ; i < model->param.nr_weight ; i++){
        weightLabels->Set(i, NanNew<Number>(model->param.weight_label[i]));
        weights->Set(i, NanNew<Number>(model->param.weight[i]));
      }
      parameters->Set(NanNew<String>("weightLabels"), weightLabels);
      parameters->Set(NanNew<String>("weighs"), weights);

      parameters->Set(NanNew<String>("cacheSize"), NanNew<Number>(model->param.cache_size));
      parameters->Set(NanNew<String>("eps"), NanNew<Number>(model->param.eps));
      if (model->param.shrinking){
        parameters->Set(NanNew<String>("shrinking"), NanTrue());
      }
      else {
        parameters->Set(NanNew<String>("shrinking"), NanFalse());
      }

      if (model->param.probability){
        parameters->Set(NanNew<String>("probability"), NanTrue());
      }
      else {
        parameters->Set(NanNew<String>("probability"), NanFalse());
      }

      obj->Set(NanNew<String>("params"), parameters);
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