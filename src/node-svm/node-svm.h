#ifndef _NODE_SVM_H
#define _NODE_SVM_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <node.h>
#include <assert.h>
#include "../../node_modules/nan/nan.h"
#include "../libsvm/svm.h"



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

        bool isTrained(){ return model != NULL;}

        bool hasParameters(){ return params != NULL;}

        bool isClassificationSVM(){
            if(!hasParameters()){
                return false;
            }

            int svm_type = params->svm_type;
            if (svm_type==NU_SVR || svm_type==EPSILON_SVR){
                return false;
            }

            return true;
        };

        bool isRegressionSVM(){ return !isClassificationSVM();};
        int getSvmType(){ return params->svm_type; };
        int getKernelType(){ return params->kernel_type; };
        int getClassNumber(){
            if(model==NULL){
                return 0;
            }
            return model->nr_class;
        }

        double getLabel(int index){ return model==NULL ? 0.0 : model->label[index]; };

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
            
            svm_params->nr_weight = 0;
            svm_params->weight_label = NULL;
            svm_params->weight = NULL;

            // check  classifer and its options
            assert(obj->Has(NanNew<String>("svmType")));
            svm_params->svm_type = obj->Get(NanNew<String>("svmType"))->IntegerValue();
            assert(svm_params->svm_type == C_SVC ||
                   svm_params->svm_type == NU_SVC ||
                   svm_params->svm_type == ONE_CLASS ||
                   svm_params->svm_type == EPSILON_SVR ||
                   svm_params->svm_type == NU_SVR);

            if (svm_params->svm_type == C_SVC ||
                svm_params->svm_type == EPSILON_SVR ||
                svm_params->svm_type == NU_SVR){
                assert(obj->Has(NanNew<String>("c")));
                svm_params->C = obj->Get(NanNew<String>("c"))->NumberValue();
            }
            if (svm_params->svm_type == NU_SVC ||
                svm_params->svm_type == NU_SVR ||
                svm_params->svm_type == ONE_CLASS){
                assert(obj->Has(NanNew<String>("nu")));
                svm_params->nu = obj->Get(NanNew<String>("nu"))->NumberValue();
                assert(svm_params->nu > 0 && 
                       svm_params->nu <= 1);
            }
            if (svm_params->svm_type == EPSILON_SVR){
                assert(obj->Has(NanNew<String>("epsilon")));
                svm_params->p = obj->Get(NanNew<String>("epsilon"))->NumberValue();
                assert(svm_params->p >= 0);
            }

            // check kernel and its options
            assert(obj->Has(NanNew<String>("kernelType")));
            svm_params->kernel_type = obj->Get(NanNew<String>("kernelType"))->IntegerValue();
            assert(svm_params->kernel_type == LINEAR ||
                   svm_params->kernel_type == POLY ||
                   svm_params->kernel_type == RBF ||
                   // svm_params->kernel_type == PRECOMPUTED ||  // not supported (yet)
                   svm_params->kernel_type == SIGMOID); 

            if (svm_params->kernel_type == POLY){
                assert(obj->Has(NanNew<String>("degree")));
                svm_params->degree = obj->Get(NanNew<String>("degree"))->IntegerValue();
                assert(svm_params->degree >= 0);
            }
            if (svm_params->kernel_type == POLY || 
                svm_params->kernel_type == RBF ||
                svm_params->kernel_type == SIGMOID){
                assert(obj->Has(NanNew<String>("gamma")));
                svm_params->gamma = obj->Get(NanNew<String>("gamma"))->NumberValue();
                assert(svm_params->gamma >= 0);
            }
            if (svm_params->kernel_type == POLY || 
                svm_params->kernel_type == SIGMOID){
                assert(obj->Has(NanNew<String>("r")));
                svm_params->coef0 = obj->Get(NanNew<String>("r"))->NumberValue();
            }

            // check training options            
            svm_params->cache_size = obj->Has(NanNew<String>("cacheSize")) ? 
                obj->Get(NanNew<String>("cacheSize"))->NumberValue() : 
                100;
            assert(svm_params->cache_size > 0);
            
            svm_params->eps = obj->Has(NanNew<String>("eps")) ? 
                obj->Get(NanNew<String>("eps"))->NumberValue() : 
                1e-3;
            assert(svm_params->eps > 0);

            svm_params->shrinking =  // enabled by default
                obj->Has(NanNew<String>("shrinking")) && 
                !obj->Get(NanNew<String>("shrinking"))->BooleanValue() ? 0 : 1;


            svm_params->probability =  // disabled by default
                obj->Has(NanNew<String>("probability")) && 
                obj->Get(NanNew<String>("probability"))->BooleanValue() ? 1 : 0;
            
            if (svm_params->svm_type == ONE_CLASS){
                assert(svm_params->probability == 0); // one-class SVM probability output not supported (yet)
            }
            
            params = svm_params;
        };

        void setModel(Local<Object> obj){
            assert(obj->Has(NanNew<String>("params")));
            assert(obj->Get(NanNew<String>("params"))->IsObject());

            setParameters(obj->Get(NanNew<String>("params"))->ToObject());
            assert(params!=NULL);
            struct svm_model *new_model = new svm_model();

            new_model->free_sv = 1;   // XXX
            new_model->rho = NULL;
            new_model->probA = NULL;
            new_model->probB = NULL;
            new_model->sv_indices = NULL;
            new_model->label = NULL;
            new_model->nSV = NULL;

            assert(obj->Has(NanNew<String>("l")));
            assert(obj->Get(NanNew<String>("l"))->IsInt32());
            new_model->l = obj->Get(NanNew<String>("l"))->IntegerValue();

            new_model->nr_class = obj->Get(NanNew<String>("nrClass"))->IntegerValue();
            unsigned int n = new_model->nr_class * (new_model->nr_class-1)/2;

            // rho
            assert(obj->Has(NanNew<String>("rho")));
            assert(obj->Get(NanNew<String>("rho"))->IsArray());
            Local<Array> rho = obj->Get(NanNew<String>("rho")).As<Array>();
            assert(rho->Length()==n);
            new_model->rho = new double[n];
            for(unsigned int i=0;i<n;i++){
                Local<Value> elt = rho->Get(i);
                assert(elt->IsNumber());
                new_model->rho[i] = elt->NumberValue();
            }

            // classes
            if (obj->Has(NanNew<String>("labels"))){
                assert(obj->Get(NanNew<String>("labels"))->IsArray());
                Local<Array> labels = obj->Get(NanNew<String>("labels")).As<Array>();
                //assert(labels->Length()==new_model->nr_class);
                new_model->label = new int[new_model->nr_class];
                for(int i=0;i<new_model->nr_class;i++){
                    Local<Value> elt = labels->Get(i);
                    assert(elt->IsInt32());
                    new_model->label[i] = elt->IntegerValue();
                }
                // nSV
                assert(obj->Has(NanNew<String>("nbSupportVectors")));
                assert(obj->Get(NanNew<String>("nbSupportVectors"))->IsArray());
                Local<Array> nbSupportVectors = obj->Get(NanNew<String>("nbSupportVectors")).As<Array>();
                assert((int)nbSupportVectors->Length() == new_model->nr_class);
                new_model->nSV = new int[new_model->nr_class];
                for (int i=0;i<new_model->nr_class;i++){
                    Local<Value> elt = nbSupportVectors->Get(i);
                    assert(elt->IsInt32());
                    new_model->nSV[i] = elt->IntegerValue();
                }
            }

            // probA
            if (obj->Has(NanNew<String>("probA"))){
                assert(obj->Get(NanNew<String>("probA"))->IsArray());
                Local<Array> probA = obj->Get(NanNew<String>("probA")).As<Array>();
                assert(probA->Length()==n);
                new_model->probA = new double[n];
                for(unsigned int i=0;i<n;i++){
                    Local<Value> elt = probA->Get(i);
                    assert(elt->IsNumber());
                    new_model->probA[i] = elt->NumberValue();
                }   
            }

            // probB
            if (obj->Has(NanNew<String>("probB"))){
                assert(obj->Get(NanNew<String>("probB"))->IsArray());
                Local<Array> probB = obj->Get(NanNew<String>("probB")).As<Array>();
                assert(probB->Length()==n);
                new_model->probB = new double[n];
                for(unsigned int i=0;i<n;i++){
                    Local<Value> elt = probB->Get(i);
                    assert(elt->IsNumber());
                    new_model->probB[i] = elt->NumberValue();
                }
            }



            // SV
            assert(obj->Has(NanNew<String>("supportVectors")));
            assert(obj->Get(NanNew<String>("supportVectors"))->IsArray());
            Local<Array> supportVectors = obj->Get(NanNew<String>("supportVectors")).As<Array>();
            assert((int)supportVectors->Length() == new_model->l);
            int m = new_model->nr_class - 1;
            int l = new_model->l;
            new_model->sv_coef = new double *[m];
            for(int i=0; i < m ;i++)
                new_model->sv_coef[i] = new double[l];
      
            new_model->SV = new svm_node*[l];
            for(int i = 0; i < l; i++) {
                Local<Array> ex = supportVectors->Get(i).As<Array>();
                assert(ex->Length()==2);
                Local<Array> x = ex->Get(0).As<Array>();
                Local<Array> y = ex->Get(1).As<Array>();

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

        void setSvmProblem(Local<Array> dataset){
            NanScope();
            struct svm_problem *prob = new svm_problem();
            prob->l = 0;

            assert(dataset->Length() > 0);    

            // check data structure and assign Y
            prob->l= dataset->Length();
            prob->y = new double[prob->l];
            int nb_features = -1;
            for (unsigned i=0; i < dataset->Length(); i++) {
                Local<Value> t = dataset->Get(i);
                assert(t->IsArray());
                Local<Array> ex = t.As<Array>();
                assert(ex->Length() == 2);

                Local<Value> tin = ex->Get(0);
                Local<Value> tout = ex->Get(1);
                assert(tin->IsArray());
                assert(tout->IsNumber());
        
                Local<Array> x = tin.As<Array>();
                if (nb_features == -1){
                    nb_features = x->Length();
                }
                else {
                    assert(nb_features == (int)x->Length()); // Incorrect dataset: all Xies should have the same length)
                }
            }

            // Asign X and Y
            prob->x = new svm_node*[dataset->Length()];
            for (unsigned i = 0; i < dataset->Length(); i++) {
                prob->x[i] = new svm_node[nb_features + 1];
                Local<Array> ex = dataset->Get(i).As<Array>();
                Local<Array> x = ex->Get(0).As<Array>();
                for (unsigned j = 0; j < x->Length(); ++j) {
                    double xi = x->Get(j)->NumberValue();
                    prob->x[i][j].index = j+1;
                    prob->x[i][j].value = xi;
                }
                prob->x[i][x->Length()].index = -1;

                double y = ex->Get(1)->NumberValue();
                prob->y[i] = y;
            }
            trainingProblem = prob;
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
            Handle<Array> supportVectors = NanNew<Array>(model->l);
            const double * const *sv_coef = model->sv_coef;
            const svm_node * const *SV = model->SV;
            const int nb_outputs = model->nr_class - 1;
            for (int i=0;i<model->l;i++){
                Handle<Array> outputs = NanNew<Array>(nb_outputs);
                for (int j=0; j < nb_outputs ; j++)
                    outputs->Set(j, NanNew<Number>(sv_coef[j][i]));

                const svm_node *p = SV[i];

                int max_index = 0;
                int nb_index = 0;
                while(p->index != -1){
                    nb_index++;
                    if (p->index > max_index)
                        max_index = p->index;
          
                    p++;
                }

                Handle<Array> inputs = NanNew<Array>(nb_index);
                int p_i = 0;
                for (int k=0; k < max_index ; k++){
                    if (k+1 == model->SV[i][p_i].index){
                        inputs->Set(k, NanNew<Number>(SV[i][p_i].value));
                        p_i++;
                    }
                    else {
                        inputs->Set(k, NanNew<Number>(0));
                    }
                }
                Handle<Array> example = NanNew<Array>(2);
                example->Set(0, inputs);
                example->Set(1, outputs);
                supportVectors->Set(i, example);
            }
            obj->Set(NanNew<String>("supportVectors"), supportVectors);

            if (model->nSV) {
                Handle<Array> nbSupportVectors = NanNew<Array>(model->nr_class);
                for(int i=0; i < model->nr_class ; i++) {
                    nbSupportVectors->Set(i, NanNew<Number>(model->nSV[i]));
                }
                obj->Set(NanNew<String>("nbSupportVectors"), nbSupportVectors);
            }
            if (model->label) {
                Handle<Array> labels = NanNew<Array>(model->nr_class);
                for (int i=0 ; i < model->nr_class ; i++){
                    labels->Set(i, NanNew<Number>(model->label[i]));
                }
                obj->Set(NanNew<String>("labels"), labels);
            }

            if (model->probA) { // regression has probA only
                int n = model->nr_class*(model->nr_class-1)/2;
                Handle<Array> probA = NanNew<Array>(n);
                for(int i=0 ; i < n ; i++){
                    probA->Set(i, NanNew<Number>(model->probA[i]));
                }
                
                obj->Set(NanNew<String>("probA"), probA);
            }

            if (model->probB) {
                int n = model->nr_class*(model->nr_class-1)/2;
                Handle<Array> probB = NanNew<Array>(n);
                for(int i=0 ; i < n ; i++){
                    probB->Set(i, NanNew<Number>(model->probB[i]));
                }
                obj->Set(NanNew<String>("probB"), probB);
            }

            if (model->rho) {
                int n = model->nr_class*(model->nr_class-1)/2;
                Handle<Array> rho = NanNew<Array>(n);
                for (int i=0 ; i < n ; i++){
                    rho->Set(i, NanNew<Number>(model->rho[i]));
                }
                obj->Set(NanNew<String>("rho"), rho);
            }

            Local<Object> parameters = NanNew<Object>();

            parameters->Set(NanNew<String>("svmType"), NanNew<Number>(model->param.svm_type));
            if (model->param.svm_type == C_SVC ||
                model->param.svm_type == EPSILON_SVR ||
                model->param.svm_type == NU_SVR){
                parameters->Set(NanNew<String>("c"), NanNew<Number>(model->param.C));
            }
            if (model->param.svm_type == NU_SVC ||
                model->param.svm_type == NU_SVR ||
                model->param.svm_type == ONE_CLASS){
                parameters->Set(NanNew<String>("nu"), NanNew<Number>(model->param.nu));
            }
            if (model->param.svm_type == EPSILON_SVR){
                parameters->Set(NanNew<String>("epsilon"), NanNew<Number>(model->param.p));
            }
            
            parameters->Set(NanNew<String>("kernelType"), NanNew<Number>(model->param.kernel_type));
            if (model->param.kernel_type == POLY){
                parameters->Set(NanNew<String>("degree"), NanNew<Number>(model->param.degree)); /* for poly */
            }

            if (model->param.kernel_type == POLY || 
                model->param.kernel_type == RBF ||
                model->param.kernel_type == SIGMOID){
                parameters->Set(NanNew<String>("gamma"), NanNew<Number>(model->param.gamma));
            }
            if (model->param.kernel_type == POLY || 
                model->param.kernel_type == SIGMOID){
                parameters->Set(NanNew<String>("r"), NanNew<Number>(model->param.coef0));
            }

            // Handle<Array> weightLabels = NanNew<Array>(model->param.nr_weight);
            // Handle<Array> weights = NanNew<Array>(model->param.nr_weight);
            // for (int i=0 ; i < model->param.nr_weight ; i++){
            //     weightLabels->Set(i, NanNew<Number>(model->param.weight_label[i]));
            //     weights->Set(i, NanNew<Number>(model->param.weight[i]));
            // }
            // parameters->Set(NanNew<String>("weightLabels"), weightLabels);
            // parameters->Set(NanNew<String>("weights"), weights);

            parameters->Set(NanNew<String>("cacheSize"), NanNew<Number>(model->param.cache_size));
            parameters->Set(NanNew<String>("eps"), NanNew<Number>(model->param.eps));
            if (model->param.shrinking == 1){
                parameters->Set(NanNew<String>("shrinking"), NanTrue());
            }
            else {
                parameters->Set(NanNew<String>("shrinking"), NanFalse());
            }

            if (model->param.probability == 1){
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