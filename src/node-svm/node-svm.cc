
#include "node-svm.h"
#include "training-worker.h"
#include "prediction-worker.h"
#include "probability-prediction-worker.h"

Persistent<Function> NodeSvm::constructor;

NodeSvm::~NodeSvm(){

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
  obj->setParameters(params);

  NanReturnUndefined();
}

NAN_METHOD(NodeSvm::Train) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->hasParameters());
  // chech params
  assert(args[0]->IsObject());

  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  obj->setSvmProblem(dataset);
  obj->train();

  NanReturnUndefined();
}

NAN_METHOD(NodeSvm::GetModel) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());

  // check obj
  assert(obj->isTrained());
  NanReturnValue(obj->getModel());
}

NAN_METHOD(NodeSvm::TrainAsync) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->hasParameters());
  // chech params
  assert(args[0]->IsObject());
  assert(args[1]->IsFunction());

  Local<Array> dataset = Array::Cast(*args[0]->ToObject());
  NanCallback *callback = new NanCallback(args[1].As<Function>());

  NanAsyncQueueWorker(new TrainingWorker(obj, dataset, callback));
  NanReturnUndefined();
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

NAN_METHOD(NodeSvm::IsTrained) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  // check obj
  NanReturnValue(Boolean::New(obj->isTrained()));
}

NAN_METHOD(NodeSvm::GetLabels) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->isTrained());

  // Create a new empty array.
  int nbClasses = obj->getClassNumber();
  Handle<Array> labels = Array::New(nbClasses);
  for (int j=0; j < nbClasses; j++){
    labels->Set(j, Number::New(obj->getLabel(j)));
  }
  NanReturnValue(labels);
}

NAN_METHOD(NodeSvm::SaveToFile) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  if (args.Length() != 1 || !args[0]->IsString())
    return ThrowException(Exception::Error(String::New("usage: NodeSvm.saveToFile(\'filename.model\'')")));
  // check obj
  assert(obj->isTrained());

  char *name = new char[4096];
  String::Cast(*args[0])->WriteAscii(name, 0, 4096);
  name[4095] = 0;

  // Create a new empty array.
  int result = obj->saveModel(name);
  delete[] name;
  NanReturnValue(Number::New(result));
}

NAN_METHOD(NodeSvm::LoadFromFile) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  if (args.Length() != 1 || !args[0]->IsString())
    return ThrowException(Exception::Error(String::New("usage: NodeSvm.loadFromFile(\'filename.model\'')")));

  char *name = new char[4096];
  String::Cast(*args[0])->WriteAscii(name, 0, 4096);
  name[4095] = 0;

  // Create a new empty array.
  obj->loadModelFromFile(name);
  delete[] name;
  NanReturnUndefined();
}

NAN_METHOD(NodeSvm::SetModel) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  if (args.Length() != 1 || !args[0]->IsObject())
    return ThrowException(Exception::Error(String::New("usage: NodeSvm.loadFromModel(model)")));

  Local<Array> model = Array::Cast(*args[0]->ToObject());
  obj->setModel(model);
  NanReturnUndefined();
}

NAN_METHOD(NodeSvm::Predict) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->isTrained());
  // chech params
  assert(args[0]->IsObject());
  
  Local<Array> inputs = Array::Cast(*args[0]->ToObject());
  if (!inputs->IsArray()){
    return ThrowException(Exception::Error(String::New("Incorrect dataset. X should be 1d-array")));
  }
  if (inputs->Length() == 0){
    return ThrowException(Exception::Error(String::New("Example data set is empty")));
  }
  svm_node *x = new svm_node[inputs->Length() + 1];
  obj->getSvmNodes(inputs, x);
  double prediction = obj->predict(x);
  delete[] x;
  NanReturnValue(Number::New(prediction));
}

NAN_METHOD(NodeSvm::PredictAsync) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->isTrained());
  // chech params
  assert(args[0]->IsObject());
  Local<Array> inputs = Array::Cast(*args[0]->ToObject());
  if (!inputs->IsArray()){
    return ThrowException(Exception::Error(String::New("Incorrect dataset. X should be 1d-array")));
  }
  if (inputs->Length() == 0){
    return ThrowException(Exception::Error(String::New("Example data set is empty")));
  }
  assert(args[1]->IsFunction());
  NanCallback *callback = new NanCallback(args[1].As<Function>());
  
  NanAsyncQueueWorker(new PredictionWorker(obj, inputs, callback));
  NanReturnUndefined();
}


NAN_METHOD(NodeSvm::PredictProbabilities) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->isTrained());
  // chech params
  assert(args[0]->IsObject());
  
  Local<Array> inputs = Array::Cast(*args[0]->ToObject());
  if (!inputs->IsArray()){
    return ThrowException(Exception::Error(String::New("Incorrect dataset. X should be 1d-array")));
  }
  if (inputs->Length() == 0){
    return ThrowException(Exception::Error(String::New("Example data set is empty")));
  }
  svm_node *x = new svm_node[inputs->Length() + 1];
  obj->getSvmNodes(inputs, x);
  
  int nbClass = obj->getClassNumber();
  double *prob_estimates = new double[nbClass];
  
  obj->predictProbabilities(x, prob_estimates);
  
  // Create the result array
  Handle<Array> probs = Array::New(nbClass);
  for (int j=0; j < nbClass; j++){
    probs->Set(j, Number::New(prob_estimates[j]));
  }
  delete[] prob_estimates;
  delete[] x;
  NanReturnValue(probs);
}

NAN_METHOD(NodeSvm::PredictProbabilitiesAsync) {
  NanScope();
  NodeSvm *obj = node::ObjectWrap::Unwrap<NodeSvm>(args.This());
  
  // check obj
  assert(obj->isTrained());
  // chech params
  assert(args[0]->IsObject());
  Local<Array> inputs = Array::Cast(*args[0]->ToObject());
  if (!inputs->IsArray()){
    return ThrowException(Exception::Error(String::New("Incorrect dataset. X should be 1d-array")));
  }
  if (inputs->Length() == 0){
    return ThrowException(Exception::Error(String::New("Example data set is empty")));
  }
  assert(args[1]->IsFunction());
  NanCallback *callback = new NanCallback(args[1].As<Function>());
  
  NanAsyncQueueWorker(new ProbabilityPredictionWorker(obj, inputs, callback));
  NanReturnUndefined();
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

  tpl->PrototypeTemplate()->Set(String::NewSymbol("trainAsync"),
      FunctionTemplate::New(NodeSvm::TrainAsync)->GetFunction());

  tpl->PrototypeTemplate()->Set(String::NewSymbol("isTrained"),
      FunctionTemplate::New(NodeSvm::IsTrained)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("getLabels"),
       FunctionTemplate::New(NodeSvm::GetLabels)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("getSvmType"),
       FunctionTemplate::New(NodeSvm::GetSvmType)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("getKernelType"),
       FunctionTemplate::New(NodeSvm::GetKernelType)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("predict"),
      FunctionTemplate::New(NodeSvm::Predict)->GetFunction());

  tpl->PrototypeTemplate()->Set(String::NewSymbol("predictAsync"),
      FunctionTemplate::New(NodeSvm::PredictAsync)->GetFunction());

  tpl->PrototypeTemplate()->Set(String::NewSymbol("predictProbabilities"),
      FunctionTemplate::New(NodeSvm::PredictProbabilities)->GetFunction());

  tpl->PrototypeTemplate()->Set(String::NewSymbol("predictProbabilitiesAsync"),
      FunctionTemplate::New(NodeSvm::PredictProbabilitiesAsync)->GetFunction());

  tpl->PrototypeTemplate()->Set(String::NewSymbol("saveToFile"),
      FunctionTemplate::New(NodeSvm::SaveToFile)->GetFunction());
  
  tpl->PrototypeTemplate()->Set(String::NewSymbol("loadFromFile"),
      FunctionTemplate::New(NodeSvm::LoadFromFile)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("loadFromModel"),
      FunctionTemplate::New(NodeSvm::SetModel)->GetFunction());

  tpl->PrototypeTemplate()->Set(String::NewSymbol("getModel"),
      FunctionTemplate::New(NodeSvm::GetModel)->GetFunction());

  constructor = Persistent<Function>::New(tpl->GetFunction());
  exports->Set(String::NewSymbol("NodeSvm"), constructor);
}