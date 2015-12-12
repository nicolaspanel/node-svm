#ifndef _NODE_SVM_H
#define _NODE_SVM_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <node.h>
#include <assert.h>
#include <nan.h>
#include "../libsvm/svm.h"



using namespace v8;

class NodeSvm : public Nan::ObjectWrap
{
    public:
        static void Init(Local<Object> exports);
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
            Local<String> svm_type_name = Nan::New<String>("svmType").ToLocalChecked();
            assert(Nan::Has(obj, svm_type_name).FromJust());
            svm_params->svm_type = Nan::Get(obj, svm_type_name).ToLocalChecked()->IntegerValue();
            assert(svm_params->svm_type == C_SVC ||
                   svm_params->svm_type == NU_SVC ||
                   svm_params->svm_type == ONE_CLASS ||
                   svm_params->svm_type == EPSILON_SVR ||
                   svm_params->svm_type == NU_SVR);

            if (svm_params->svm_type == C_SVC ||
                svm_params->svm_type == EPSILON_SVR ||
                svm_params->svm_type == NU_SVR){

                Local<String> str_c = Nan::New<String>("c").ToLocalChecked();
                assert(Nan::Has(obj, str_c).FromJust());
                svm_params->C = Nan::Get(obj, str_c).ToLocalChecked()->NumberValue();
            }
            if (svm_params->svm_type == NU_SVC ||
                svm_params->svm_type == NU_SVR ||
                svm_params->svm_type == ONE_CLASS){


                Local<String> str_nu = Nan::New<String>("nu").ToLocalChecked();
                assert(Nan::Has(obj, str_nu).FromJust());
                svm_params->nu = Nan::Get(obj, str_nu).ToLocalChecked()->NumberValue();
                assert(svm_params->nu > 0 &&
                       svm_params->nu <= 1);
            }
            if (svm_params->svm_type == EPSILON_SVR){

                Local<String> str_epsilon = Nan::New<String>("epsilon").ToLocalChecked();
                assert(Nan::Has(obj, str_epsilon).FromJust());
                svm_params->p = Nan::Get(obj, str_epsilon).ToLocalChecked()->NumberValue();
                assert(svm_params->p >= 0);
            }

            // check kernel and its options
            Local<String> str_kernel_type = Nan::New<String>("kernelType").ToLocalChecked();
            assert(Nan::Has(obj, str_kernel_type).FromJust());
            svm_params->kernel_type = Nan::Get(obj, str_kernel_type).ToLocalChecked()->IntegerValue();
            assert(svm_params->kernel_type == LINEAR ||
                   svm_params->kernel_type == POLY ||
                   svm_params->kernel_type == RBF ||
                   // svm_params->kernel_type == PRECOMPUTED ||  // not supported (yet)
                   svm_params->kernel_type == SIGMOID);

            if (svm_params->kernel_type == POLY){

                Local<String> str_degree = Nan::New<String>("degree").ToLocalChecked();

                assert(Nan::Has(obj, str_degree).FromJust());
                svm_params->degree = Nan::Get(obj, str_degree).ToLocalChecked()->IntegerValue();
                assert(svm_params->degree >= 0);
            }
            if (svm_params->kernel_type == POLY ||
                svm_params->kernel_type == RBF ||
                svm_params->kernel_type == SIGMOID){

                Local<String> str_gamma = Nan::New<String>("gamma").ToLocalChecked();

                assert(Nan::Has(obj, str_gamma).FromJust());
                svm_params->gamma = Nan::Get(obj, str_gamma).ToLocalChecked()->NumberValue();
                assert(svm_params->gamma >= 0);
            }
            if (svm_params->kernel_type == POLY ||
                svm_params->kernel_type == SIGMOID){

                Local<String> str_r = Nan::New<String>("r").ToLocalChecked();

                assert(Nan::Has(obj, str_r).FromJust());
                svm_params->coef0 = Nan::Get(obj, str_r).ToLocalChecked()->NumberValue();
            }

            // check training options
            Local<String> str_cache_size = Nan::New<String>("cacheSize").ToLocalChecked();
            svm_params->cache_size = Nan::Has(obj, str_cache_size).FromJust() ?
                Nan::Get(obj, str_cache_size).ToLocalChecked()->NumberValue() :
                100;
            assert(svm_params->cache_size > 0);

            Local<String> str_eps = Nan::New<String>("eps").ToLocalChecked();
            svm_params->eps = Nan::Has(obj, str_eps).FromJust() ?
                Nan::Get(obj, str_eps).ToLocalChecked()->NumberValue() :
                1e-3;
            assert(svm_params->eps > 0);

            Local<String> str_shrinking = Nan::New<String>("shrinking").ToLocalChecked();
            svm_params->shrinking =  // enabled by default
                Nan::Has(obj, str_shrinking).FromJust() &&
                !Nan::Get(obj, str_shrinking).ToLocalChecked()->BooleanValue() ? 0 : 1;


            Local<String> str_probability = Nan::New<String>("probability").ToLocalChecked();
            svm_params->probability =  // disabled by default
                Nan::Has(obj, str_probability).FromJust() &&
                Nan::Get(obj, str_probability).ToLocalChecked()->BooleanValue() ? 1 : 0;

            if (svm_params->svm_type == ONE_CLASS){
                assert(svm_params->probability == 0); // one-class SVM probability output not supported (yet)
            }

            params = svm_params;
        };

        void setModel(Local<Object> obj){
            Local<String> str_params = Nan::New<String>("params").ToLocalChecked();

            assert(Nan::Has(obj, str_params).FromJust());
            assert(Nan::Get(obj, str_params).ToLocalChecked()->IsObject());

            setParameters(Nan::Get(obj, str_params).ToLocalChecked()->ToObject());
            assert(params!=NULL);
            struct svm_model *new_model = new svm_model();

            new_model->free_sv = 1;   // XXX
            new_model->rho = NULL;
            new_model->probA = NULL;
            new_model->probB = NULL;
            new_model->sv_indices = NULL;
            new_model->label = NULL;
            new_model->nSV = NULL;

            Local<String> str_l = Nan::New<String>("l").ToLocalChecked();

            assert(Nan::Has(obj, str_l).FromJust());
            assert(Nan::Get(obj, str_l).ToLocalChecked()->IsInt32());
            new_model->l = Nan::Get(obj, str_l).ToLocalChecked()->IntegerValue();

            Local<String> str_nr_class = Nan::New<String>("nrClass").ToLocalChecked();

            new_model->nr_class = Nan::Get(obj, str_nr_class).ToLocalChecked()->IntegerValue();
            unsigned int n = new_model->nr_class * (new_model->nr_class-1)/2;

            // rho
            Local<String> str_rho = Nan::New<String>("rho").ToLocalChecked();

            assert(Nan::Has(obj, str_rho).FromJust());
            assert(Nan::Get(obj, str_rho).ToLocalChecked()->IsArray());
            Local<Array> rho = Nan::Get(obj, str_rho).ToLocalChecked().As<Array>();
            assert(rho->Length()==n);
            new_model->rho = new double[n];
            for(unsigned int i=0;i<n;i++){
                Local<Value> elt = rho->Get(i);
                assert(elt->IsNumber());
                new_model->rho[i] = elt->NumberValue();
            }

            // classes
            Local<String> str_labels = Nan::New<String>("labels").ToLocalChecked();
            if (Nan::Has(obj, str_labels).FromJust()){
                assert(Nan::Get(obj, str_labels).ToLocalChecked()->IsArray());
                Local<Array> labels = Nan::Get(obj, str_labels).ToLocalChecked().As<Array>();
                //assert(labels->Length()==new_model->nr_class);
                new_model->label = new int[new_model->nr_class];
                for(int i=0;i<new_model->nr_class;i++){
                    Local<Value> elt = labels->Get(i);
                    assert(elt->IsInt32());
                    new_model->label[i] = elt->IntegerValue();
                }
                // nSV

                Local<String> str_nb_support_vectors = Nan::New<String>("nbSupportVectors").ToLocalChecked();
                assert(Nan::Has(obj, str_nb_support_vectors).FromJust());
                assert(Nan::Get(obj, str_nb_support_vectors).ToLocalChecked()->IsArray());
                Local<Array> nbSupportVectors = Nan::Get(obj, str_nb_support_vectors).ToLocalChecked().As<Array>();
                assert((int)nbSupportVectors->Length() == new_model->nr_class);
                new_model->nSV = new int[new_model->nr_class];
                for (int i=0;i<new_model->nr_class;i++){
                    Local<Value> elt = nbSupportVectors->Get(i);
                    assert(elt->IsInt32());
                    new_model->nSV[i] = elt->IntegerValue();
                }
            }

            // probA
            Local<String> str_prob_a = Nan::New<String>("probA").ToLocalChecked();
            if (Nan::Has(obj, str_prob_a).FromJust()){
                assert(Nan::Get(obj, str_prob_a).ToLocalChecked()->IsArray());
                Local<Array> probA = Nan::Get(obj, str_prob_a).ToLocalChecked().As<Array>();
                assert(probA->Length()==n);
                new_model->probA = new double[n];
                for(unsigned int i=0;i<n;i++){
                    Local<Value> elt = probA->Get(i);
                    assert(elt->IsNumber());
                    new_model->probA[i] = elt->NumberValue();
                }
            }

            // probB
            Local<String> str_prob_b = Nan::New<String>("probB").ToLocalChecked();
            if (Nan::Has(obj, str_prob_b).FromJust()){
                assert(Nan::Get(obj, str_prob_b).ToLocalChecked()->IsArray());
                Local<Array> probB = Nan::Get(obj, str_prob_b).ToLocalChecked().As<Array>();
                assert(probB->Length()==n);
                new_model->probB = new double[n];
                for(unsigned int i=0;i<n;i++){
                    Local<Value> elt = probB->Get(i);
                    assert(elt->IsNumber());
                    new_model->probB[i] = elt->NumberValue();
                }
            }



            // SV
            Local<String> str_support_vectors = Nan::New<String>("supportVectors").ToLocalChecked();

            assert(Nan::Has(obj, str_support_vectors).FromJust());
            assert(Nan::Get(obj, str_support_vectors).ToLocalChecked()->IsArray());
            Local<Array> supportVectors = Nan::Get(obj, str_support_vectors).ToLocalChecked().As<Array>();
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
            Nan::HandleScope scope;
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
            Local<Object> obj = Nan::New<Object>();
            Local<String> str_nr_class = Nan::New<String>("nrClass").ToLocalChecked();
            Local<String> str_l = Nan::New<String>("l").ToLocalChecked();
            obj->Set(str_nr_class, Nan::New<Number>(model->nr_class));
            obj->Set(str_l, Nan::New<Number>(model->l));

            // Create a new array for support vectors
            Local<Array> supportVectors = Nan::New<Array>(model->l);
            const double * const *sv_coef = model->sv_coef;
            const svm_node * const *SV = model->SV;
            const int nb_outputs = model->nr_class - 1;
            for (int i=0;i<model->l;i++){
                Local<Array> outputs = Nan::New<Array>(nb_outputs);
                for (int j=0; j < nb_outputs ; j++)
                    outputs->Set(j, Nan::New<Number>(sv_coef[j][i]));

                const svm_node *p = SV[i];

                int max_index = 0;
                int nb_index = 0;
                while(p->index != -1){
                    nb_index++;
                    if (p->index > max_index)
                        max_index = p->index;

                    p++;
                }

                Local<Array> inputs = Nan::New<Array>(nb_index);
                int p_i = 0;
                for (int k=0; k < max_index ; k++){
                    if (k+1 == model->SV[i][p_i].index){
                        inputs->Set(k, Nan::New<Number>(SV[i][p_i].value));
                        p_i++;
                    }
                    else {
                        inputs->Set(k, Nan::New<Number>(0));
                    }
                }
                Local<Array> example = Nan::New<Array>(2);
                example->Set(0, inputs);
                example->Set(1, outputs);
                supportVectors->Set(i, example);
            }
            Local<String> str_support_vectors = Nan::New<String>("supportVectors").ToLocalChecked();
            obj->Set(str_support_vectors, supportVectors);

            if (model->nSV) {
                Local<Array> nbSupportVectors = Nan::New<Array>(model->nr_class);
                for(int i=0; i < model->nr_class ; i++) {
                    nbSupportVectors->Set(i, Nan::New<Number>(model->nSV[i]));
                }

                Local<String> str_nb_support_vectors = Nan::New<String>("nbSupportVectors").ToLocalChecked();
                obj->Set(str_nb_support_vectors, nbSupportVectors);
            }

            if (model->label) {
                Local<Array> labels = Nan::New<Array>(model->nr_class);
                for (int i=0 ; i < model->nr_class ; i++){
                    labels->Set(i, Nan::New<Number>(model->label[i]));
                }
                Local<String> str_labels = Nan::New<String>("labels").ToLocalChecked();
                obj->Set(str_labels, labels);
            }

            if (model->probA) { // regression has probA only
                int n = model->nr_class*(model->nr_class-1)/2;
                Local<Array> probA = Nan::New<Array>(n);
                for(int i=0 ; i < n ; i++){
                    probA->Set(i, Nan::New<Number>(model->probA[i]));
                }

                Local<String> str_prob_a = Nan::New<String>("probA").ToLocalChecked();
                obj->Set(str_prob_a, probA);
            }

            if (model->probB) {
                int n = model->nr_class*(model->nr_class-1)/2;
                Local<Array> probB = Nan::New<Array>(n);
                for(int i=0 ; i < n ; i++){
                    probB->Set(i, Nan::New<Number>(model->probB[i]));
                }

                Local<String> str_prob_b = Nan::New<String>("probB").ToLocalChecked();
                obj->Set(str_prob_b, probB);
            }

            if (model->rho) {
                int n = model->nr_class*(model->nr_class-1)/2;
                Local<Array> rho = Nan::New<Array>(n);
                for (int i=0 ; i < n ; i++){
                    rho->Set(i, Nan::New<Number>(model->rho[i]));
                }

                Local<String> str_rho = Nan::New<String>("rho").ToLocalChecked();
                obj->Set(str_rho, rho);
            }

            Local<Object> parameters = Nan::New<Object>();

            Local<String> str_svm_type = Nan::New<String>("svmType").ToLocalChecked();
            parameters->Set(str_svm_type, Nan::New<Number>(model->param.svm_type));
            if (model->param.svm_type == C_SVC ||
                model->param.svm_type == EPSILON_SVR ||
                model->param.svm_type == NU_SVR){

                Local<String> str_c = Nan::New<String>("c").ToLocalChecked();
                parameters->Set(str_c, Nan::New<Number>(model->param.C));
            }

            if (model->param.svm_type == NU_SVC ||
                model->param.svm_type == NU_SVR ||
                model->param.svm_type == ONE_CLASS){

                Local<String> str_nu = Nan::New<String>("nu").ToLocalChecked();
                parameters->Set(str_nu, Nan::New<Number>(model->param.nu));
            }

            if (model->param.svm_type == EPSILON_SVR){
                Local<String> str_epsilon = Nan::New<String>("epsilon").ToLocalChecked();
                parameters->Set(str_epsilon, Nan::New<Number>(model->param.p));
            }

            Local<String> str_kernel_type = Nan::New<String>("kernelType").ToLocalChecked();
            parameters->Set(str_kernel_type, Nan::New<Number>(model->param.kernel_type));
            if (model->param.kernel_type == POLY){
                Local<String> str_degree = Nan::New<String>("degree").ToLocalChecked();
                parameters->Set(str_degree, Nan::New<Number>(model->param.degree)); /* for poly */
            }

            if (model->param.kernel_type == POLY ||
                model->param.kernel_type == RBF ||
                model->param.kernel_type == SIGMOID){
                Local<String> str_gamma = Nan::New<String>("gamma").ToLocalChecked();
                parameters->Set(str_gamma, Nan::New<Number>(model->param.gamma));
            }
            if (model->param.kernel_type == POLY ||
                model->param.kernel_type == SIGMOID){
                Local<String> str_r = Nan::New<String>("r").ToLocalChecked();
                parameters->Set(str_r, Nan::New<Number>(model->param.coef0));
            }

            // Handle<Array> weightLabels = NanNew<Array>(model->param.nr_weight);
            // Handle<Array> weights = NanNew<Array>(model->param.nr_weight);
            // for (int i=0 ; i < model->param.nr_weight ; i++){
            //     weightLabels->Set(i, NanNew<Number>(model->param.weight_label[i]));
            //     weights->Set(i, NanNew<Number>(model->param.weight[i]));
            // }
            // parameters->Set(NanNew<String>("weightLabels"), weightLabels);
            // parameters->Set(NanNew<String>("weights"), weights);


            Local<String> str_cache_size = Nan::New<String>("cacheSize").ToLocalChecked();
            Local<String> str_eps = Nan::New<String>("eps").ToLocalChecked();
            parameters->Set(str_cache_size, Nan::New<Number>(model->param.cache_size));
            parameters->Set(str_eps, Nan::New<Number>(model->param.eps));

            Local<String> str_shrinking = Nan::New<String>("shrinking").ToLocalChecked();
            if (model->param.shrinking == 1){
                parameters->Set(str_shrinking, Nan::True());
            }
            else {
                parameters->Set(str_shrinking, Nan::False());
            }

            Local<String> str_probability = Nan::New<String>("probability").ToLocalChecked();
            if (model->param.probability == 1){
                parameters->Set(str_probability, Nan::True());
            }
            else {
                parameters->Set(str_probability, Nan::False());
            }

            Local<String> str_params = Nan::New<String>("params").ToLocalChecked();
            obj->Set(str_params, parameters);
            return obj;
        };
  private:
    ~NodeSvm();
    struct svm_parameter *params;
    struct svm_model *model;
    struct svm_problem *trainingProblem;
    static Nan::Persistent<Function> constructor;

};

#endif /* _NODE_SVM_H */
