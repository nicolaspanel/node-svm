
#include "node-svm.h"
#include "training-worker.h"
#include "prediction-worker.h"
#include "probability-prediction-worker.h"

using v8::FunctionTemplate;
using v8::Object;
using v8::String;
using v8::Array;

Nan::Persistent<Function> NodeSvm::constructor;

NodeSvm::~NodeSvm(){

}

NAN_METHOD(NodeSvm::New) {
    Nan::HandleScope scope;

    if (info.IsConstructCall()) {
        // Invoked as constructor: `new MyObject(...)`
        NodeSvm* obj = new NodeSvm();
        obj->Wrap(info.This());
        info.GetReturnValue().Set(info.This());
    }
    else {
        // Invoked as plain function `MyObject(...)`, turn into construct call.
        const int argc = 0;
#ifdef _WIN32
    // On windows you get "error C2466: cannot allocate an array of constant size 0" and we use a pointer
    Local<Value>* argv;
#else
    Local<Value> argv[argc];
#endif
        Local<Function> cons = Nan::New<Function>(constructor);
        info.GetReturnValue().Set(cons->NewInstance(argc, argv));
    }
}

NAN_METHOD(NodeSvm::SetParameters) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    assert(info[0]->IsObject());
    Local<Object> params = info[0].As<Object>();
    obj->setParameters(params);
}

NAN_METHOD(NodeSvm::Train) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    // check obj
    assert(obj->hasParameters());
    // chech params
    assert(info[0]->IsObject());

    Local<Array> dataset = info[0].As<Array>();
    obj->setSvmProblem(dataset);
    obj->train();
}

NAN_METHOD(NodeSvm::GetModel) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    // check obj
    assert(obj->isTrained());
    info.GetReturnValue().Set(obj->getModel());
}

NAN_METHOD(NodeSvm::TrainAsync) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    // check obj
    assert(obj->hasParameters());
    // chech params
    assert(info[0]->IsObject());
    assert(info[1]->IsFunction());

    Local<Array> dataset = info[0].As<Array>();
    Nan::Callback *callback = new Nan::Callback(info[1].As<Function>());

    Nan::AsyncQueueWorker(new TrainingWorker(obj, dataset, callback));
}

NAN_METHOD(NodeSvm::GetKernelType) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());
    // check obj
    assert(obj->hasParameters());
    info.GetReturnValue().Set(Nan::New<Number>(obj->getKernelType()));
}

NAN_METHOD(NodeSvm::GetSvmType) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());
    // check obj
    assert(obj->hasParameters());
    info.GetReturnValue().Set(Nan::New<Number>(obj->getSvmType()));
}

NAN_METHOD(NodeSvm::IsTrained) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());
    // check obj
    info.GetReturnValue().Set(Nan::New<Boolean>(obj->isTrained()));
}

NAN_METHOD(NodeSvm::GetLabels) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    // check obj
    assert(obj->isTrained());

    // Create a new empty array.
    int nbClasses = obj->getClassNumber();
    Local<Array> labels = Nan::New<Array>(nbClasses);
    for (int j=0; j < nbClasses; j++){
        labels->Set(j, Nan::New<Number>(obj->getLabel(j)));
    }
    info.GetReturnValue().Set(labels);
}

NAN_METHOD(NodeSvm::SetModel) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());
    assert(info.Length() == 1);
    assert(info[0]->IsObject());

    Local<Array> model = info[0].As<Array>();
    obj->setModel(model);
}

NAN_METHOD(NodeSvm::Predict) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    // check obj
    assert(obj->isTrained());
    // chech params
    assert(info[0]->IsObject());

    Local<Array> inputs = info[0].As<Array>();
    assert(inputs->IsArray());
    assert(inputs->Length() > 0);
    svm_node *x = new svm_node[inputs->Length() + 1];
    obj->getSvmNodes(inputs, x);
    double prediction = obj->predict(x);
    delete[] x;
    info.GetReturnValue().Set(Nan::New<Number>(prediction));
}

NAN_METHOD(NodeSvm::PredictAsync) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    // check obj
    assert(obj->isTrained());
    // chech params
    assert(info[0]->IsObject());
    Local<Array> inputs = info[0].As<Array>();
    assert(inputs->IsArray());
    assert(inputs->Length() > 0);
    assert(info[1]->IsFunction());
    Nan::Callback *callback = new Nan::Callback(info[1].As<Function>());

    Nan::AsyncQueueWorker(new PredictionWorker(obj, inputs, callback));
}


NAN_METHOD(NodeSvm::PredictProbabilities) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    // check obj
    assert(obj->isTrained());
    // chech params
    assert(info[0]->IsObject());

    Local<Array> inputs = info[0].As<Array>();
    assert(inputs->IsArray());
    assert(inputs->Length() > 0);
    svm_node *x = new svm_node[inputs->Length() + 1];
    obj->getSvmNodes(inputs, x);

    int nbClass = obj->getClassNumber();
    double *prob_estimates = new double[nbClass];

    obj->predictProbabilities(x, prob_estimates);

    // Create the result array
    Local<Array> probs = Nan::New<Array>(nbClass);
    for (int j=0; j < nbClass; j++){
        probs->Set(j, Nan::New<Number>(prob_estimates[j]));
    }
    delete[] prob_estimates;
    delete[] x;
    info.GetReturnValue().Set(probs);
}

NAN_METHOD(NodeSvm::PredictProbabilitiesAsync) {
    Nan::HandleScope scope;
    NodeSvm *obj = Nan::ObjectWrap::Unwrap<NodeSvm>(info.This());

    // check obj
    assert(obj->isTrained());
    // chech params
    assert(info[0]->IsObject());
    Local<Array> inputs = info[0].As<Array>();
    assert(inputs->IsArray());
    assert(inputs->Length() > 0);
    assert(info[1]->IsFunction());
    Nan::Callback *callback = new Nan::Callback(info[1].As<Function>());

    Nan::AsyncQueueWorker(new ProbabilityPredictionWorker(obj, inputs, callback));
}

void NodeSvm::Init(Local<Object> exports){
    // Prepare constructor template
    Local<FunctionTemplate> tpl = Nan::New<FunctionTemplate>(NodeSvm::New);
    tpl->SetClassName(Nan::New<String>("NodeSvm").ToLocalChecked());
    tpl->InstanceTemplate()->SetInternalFieldCount(1);

    // prototype
    tpl->PrototypeTemplate()->Set(Nan::New<String>("setParameters").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::SetParameters));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("train").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::Train));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("trainAsync").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::TrainAsync));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("isTrained").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::IsTrained));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("getLabels").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::GetLabels));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("getSvmType").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::GetSvmType));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("getKernelType").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::GetKernelType));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("predict").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::Predict));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("predictAsync").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::PredictAsync));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("predictProbabilities").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::PredictProbabilities));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("predictProbabilitiesAsync").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::PredictProbabilitiesAsync));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("loadFromModel").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::SetModel));

    tpl->PrototypeTemplate()->Set(Nan::New<String>("getModel").ToLocalChecked(),
    Nan::New<FunctionTemplate>(NodeSvm::GetModel));

    //constructor = Persistent<Function>::New(tpl->GetFunction());
    exports->Set(Nan::New<String>("NodeSvm").ToLocalChecked(), tpl->GetFunction());
    constructor.Reset(Nan::GetFunction(tpl).ToLocalChecked());
}
