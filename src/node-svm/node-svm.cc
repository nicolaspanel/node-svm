
#include "node-svm.h"
#include "training-job.h"

Persistent<Function> NodeSvm::constructor;

NodeSvm::~NodeSvm() {
}

NAN_METHOD(NodeSvm::New) {
  NanScope();

  if (args.IsConstructCall()) {
    // Invoked as constructor: `new MyObject(...)`
    NodeSvm* obj = new NodeSvm();
    obj->Wrap(args.This());
    return args.This();
  } else {
    // Invoked as plain function `MyObject(...)`, turn into construct call.
    const int argc = 0;
    Local<Value> argv[argc] = {  };
    return scope.Close(constructor->NewInstance(argc, argv));
  }
}

NAN_METHOD(NodeSvm::SetParameters) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  assert(args[0]->IsObject());
  Local<Object> parameters = *args[0]->ToObject();
  svm_parameter *svm_params = new svm_parameter();
  svm_params->svm_type = parameters->Get(String::New("type"))->IntegerValue();
  svm_params->kernel_type = parameters->Get(String::New("kernel"))->IntegerValue();
  svm_params->degree = parameters->Get(String::New("degree"))->IntegerValue();
  svm_params->gamma = parameters->Get(String::New("gamma"))->NumberValue();
  svm_params->coef0 = parameters->Get(String::New("r"))->NumberValue();
  
  svm_params->cache_size = parameters->Get(String::New("cacheSize"))->NumberValue();
  svm_params->eps = parameters->Get(String::New("eps"))->NumberValue();
  svm_params->C = parameters->Get(String::New("C"))->NumberValue();
  svm_params->nu = parameters->Get(String::New("nu"))->NumberValue();
  svm_params->p = parameters->Get(String::New("p"))->NumberValue();
  svm_params->shrinking = parameters->Get(String::New("shrinking"))->IntegerValue();
  svm_params->probability = parameters->Get(String::New("probability"))->IntegerValue();

  obj->params = svm_params;
  svm_problem *svm_prob = new svm_problem();
  svm_prob->l = 0;
  const char *error_msg;
  error_msg =  svm_check_parameter(svm_prob, svm_params);
  Local<String> error = String::New("");
  if (error_msg)
    error = String::New("Invalid parameters");
  NanReturnValue(error);
}

NAN_METHOD(NodeSvm::Train) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  assert(args[0]->IsObject());
  assert(args[1]->IsFunction());
  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  NanCallback *callback = new NanCallback(args[1].As<Function>());
  NanAsyncQueueWorker(new TrainingJob(dataset, obj, callback));
  NanReturnUndefined();
}

NAN_METHOD(NodeSvm::GetLabels) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  assert(obj->model != NULL);
  // Create a new empty array.
  Handle<Array> labels = Array::New(obj->model->nr_class);
  for (int j=0; j < obj->model->nr_class; j++){
    labels->Set(j, Number::New(obj->model->label[j]));
  }
  NanReturnValue(labels);
}

NAN_METHOD(NodeSvm::Predict) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  assert(args[0]->IsObject());
  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  svm_node *x = Malloc(struct svm_node,dataset->Length());
  for (unsigned j=0; j < dataset->Length(); j++){
    x[j].index = j;
    x[j].value = dataset->Get(j)->NumberValue();
  }
  double prediction = svm_predict(obj->model, x);
  NanReturnValue(Number::New(prediction));
}

NAN_METHOD(NodeSvm::PredictAsync) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  assert(args[0]->IsObject());
  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  svm_node *x = Malloc(struct svm_node,dataset->Length());
  for (unsigned j=0; j < dataset->Length(); j++){
    x[j].index = j;
    x[j].value = dataset->Get(j)->NumberValue();
  }
  assert(args[1]->IsFunction());
  NanCallback *callback = new NanCallback(args[1].As<Function>());

  double prediction = svm_predict(obj->model, x);
  Local<Value> argv[] = {
    Number::New(prediction)
  };
  callback->Call(1, argv);
  NanReturnUndefined();
}


NAN_METHOD(NodeSvm::PredictProbabilities) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  assert(args[0]->IsObject());
  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  svm_node *x = Malloc(struct svm_node,dataset->Length());
  for (unsigned j=0; j < dataset->Length(); j++){
    x[j].index = j;
    x[j].value = dataset->Get(j)->NumberValue();
  }
  double *prob_estimates = Malloc(double,obj->model->nr_class);
  svm_predict_probability(obj->model,x,prob_estimates);
  // Create a new empty array.
  Handle<Array> probs = Array::New(obj->model->nr_class);
  for (int j=0; j < obj->model->nr_class; j++){
    probs->Set(j, Number::New(prob_estimates[j]));
  }
  NanReturnValue(probs);
}

void NodeSvm::trainInstance(svm_problem *problem){
  model = svm_train(problem, params);
}

void NodeSvm::Init(Handle<Object> exports){
  // Prepare constructor template
  Local<FunctionTemplate> tpl = FunctionTemplate::New(NodeSvm::New);
  tpl->SetClassName(String::NewSymbol("NodeSvm"));
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  // prototype
  tpl->PrototypeTemplate()->Set(String::NewSymbol("setParameters"),
      FunctionTemplate::New(NodeSvm::SetParameters)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("train"),
      FunctionTemplate::New(NodeSvm::Train)->GetFunction());
   tpl->PrototypeTemplate()->Set(String::NewSymbol("getLabels"),
       FunctionTemplate::New(NodeSvm::GetLabels)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("predict"),
      FunctionTemplate::New(NodeSvm::Predict)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("predictAsync"),
      FunctionTemplate::New(NodeSvm::PredictAsync)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("predictProbabilities"),
      FunctionTemplate::New(NodeSvm::PredictProbabilities)->GetFunction());

  constructor = Persistent<Function>::New(tpl->GetFunction());
  exports->Set(String::NewSymbol("NodeSvm"), constructor);
}