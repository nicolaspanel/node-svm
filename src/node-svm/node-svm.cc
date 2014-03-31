
#include "node-svm.h"
#include "accuracy-job.h"

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
  Local<Object> params = *args[0]->ToObject();
  svm_parameter *svm_params = new svm_parameter();
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

  if (params->Has(String::New("type"))){
    svm_params->svm_type = params->Get(String::New("type"))->IntegerValue();
  }
  if (params->Has(String::New("kernel"))){
    svm_params->kernel_type = params->Get(String::New("kernel"))->IntegerValue();
  }
  if (params->Has(String::New("degree"))){
    svm_params->degree = params->Get(String::New("degree"))->IntegerValue();
  }
  if (params->Has(String::New("gamma"))){
    svm_params->gamma = params->Get(String::New("gamma"))->NumberValue();
  }
  if (params->Has(String::New("r"))){
    svm_params->coef0 = params->Get(String::New("r"))->NumberValue();
  }
  if (params->Has(String::New("C"))){
    svm_params->C = params->Get(String::New("C"))->NumberValue();
  }
  if (params->Has(String::New("nu"))){
    svm_params->nu = params->Get(String::New("nu"))->NumberValue();
  }
  if (params->Has(String::New("p"))){
    svm_params->p = params->Get(String::New("p"))->NumberValue();
  }
  
  if (params->Has(String::New("cacheSize"))){
    svm_params->cache_size = params->Get(String::New("cacheSize"))->NumberValue();
  }
  if (params->Has(String::New("eps"))){
    svm_params->eps = params->Get(String::New("eps"))->NumberValue();
  }
  if (params->Has(String::New("shrinking"))){
    svm_params->shrinking = params->Get(String::New("shrinking"))->IntegerValue();
  }
  if (params->Has(String::New("probability"))){
    svm_params->probability = params->Get(String::New("probability"))->IntegerValue();
  }
  

  obj->params = svm_params;
  svm_problem *svm_prob = new svm_problem();
  svm_prob->l = 0;
  const char *error_msg;
  error_msg =  svm_check_parameter(svm_prob, svm_params);
  std::cout << error_msg << std::endl;
  Local<String> error = String::New("");
  if (error_msg)
    error = String::New("Invalid parameters");
  NanReturnValue(error);
}

NAN_METHOD(NodeSvm::Train) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->hasParameters());
  // chech params
  assert(args[0]->IsObject());

  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  assert(dataset->Length() > 0);
  struct svm_problem *prob = libsvm::convert_data_to_problem(dataset);
  assert(prob->l > 0);

  svm_model *model = svm_train(prob, obj->params);
  svm_save_model("test2.model", model);
  obj->model = model;
  
  Local<Value> argv[] = {
    
  };
  NanReturnValue(String::New("ko"));
}

NAN_METHOD(NodeSvm::GetKernelType) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  // check obj
  assert(obj->hasParameters());
  NanReturnValue(Number::New(obj->getKernelType()));
}

NAN_METHOD(NodeSvm::GetSvmType) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  // check obj
  assert(obj->hasParameters());
  NanReturnValue(Number::New(obj->getSvmType()));
}

NAN_METHOD(NodeSvm::GetAccuracy) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  // check obj
  assert(obj->isTrained());
  assert(obj->isClassificationSVM());
  // chech params
  assert(args[0]->IsObject());
  assert(args[1]->IsFunction());
  
  Local<Array> testset = Array::Cast(*args[0]->ToObject());
  struct svm_problem *prob = libsvm::convert_data_to_problem(testset);
  NanCallback *callback = new NanCallback(args[1].As<Function>());
  NanAsyncQueueWorker(new AccuracyJob(prob, obj->model, callback));
  NanReturnUndefined();
}

NAN_METHOD(NodeSvm::GetLabels) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->isTrained());

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
  
  // check obj
  assert(obj->isTrained());
  // chech params
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
  
  // check obj
  assert(obj->isTrained());
  // chech params
  assert(args[0]->IsObject());
  assert(args[1]->IsFunction());

  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  svm_node *x = Malloc(struct svm_node,dataset->Length());
  for (unsigned j=0; j < dataset->Length(); j++){
    x[j].index = j;
    x[j].value = dataset->Get(j)->NumberValue();
  }
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
  
  // check obj
  assert(obj->isTrained());
  // chech params
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
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("getSvmType"),
       FunctionTemplate::New(NodeSvm::GetSvmType)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("getKernelType"),
       FunctionTemplate::New(NodeSvm::GetKernelType)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("getAccuracy"),
       FunctionTemplate::New(NodeSvm::GetAccuracy)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("predict"),
      FunctionTemplate::New(NodeSvm::Predict)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("predictAsync"),
      FunctionTemplate::New(NodeSvm::PredictAsync)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("predictProbabilities"),
      FunctionTemplate::New(NodeSvm::PredictProbabilities)->GetFunction());

  constructor = Persistent<Function>::New(tpl->GetFunction());
  exports->Set(String::NewSymbol("NodeSvm"), constructor);
}