
#include "node-svm.h"

Handle<Value> VException(const char *msg)
{
  NanScope();
  return ThrowException(Exception::Error(String::New(msg)));
}

Handle<Value> getSvmProblemFromData(Local<Array> dataset, svm_problem *prob){
  NanScope();
  prob->l = 0;

  if (dataset->Length() == 0)
    return VException("Training data set is empty");
  else
    fprintf(stderr,"problem contains %d examples\n",dataset->Length());
  

  // check data structure and assign Y
  prob->l= dataset->Length();
  prob->y = new double[prob->l];
  int *nb_features = new int[dataset->Length()];
  for (unsigned i=0; i < dataset->Length(); i++) {
    Local<Value> t = dataset->Get(i);
    if (!t->IsArray()) {
      return VException("First argument should be 2d-array (training data set)");
    }
    Local<Array> ex = Array::Cast(*t);
    if (ex->Length() != 2) {
      return VException("Incorrect dataset (should be composed of two columns)");
    }

    Local<Value> tin = ex->Get(0);
    Local<Value> tout = ex->Get(1);
    if (!tin->IsArray() || !tout->IsNumber()) {
      return VException("Incorrect dataset (first column should be an array and second column should be a number)");
    }
    Local<Array> x = Array::Cast(*tin);
    double y = tout->NumberValue();
    prob->y[i] = y;
    fprintf(stderr,"%g ",y);
    int n = 0;
    for (unsigned j=0; j < x->Length(); j++) {
      double xi = x->Get(j)->NumberValue();
      fprintf(stderr,"%d:%g ",j+1, xi);
      if (xi!=0){
        n++;
      }
    }
    fprintf(stderr,"(%d xies)\n",n);
    nb_features[i] = n;
  }

  // Asign X
  prob->x = new svm_node*[dataset->Length()];
  for(unsigned i = 0; i < dataset->Length(); i++) {
    prob->x[i] = new svm_node[nb_features[i]];
    Local<Array> ex = Array::Cast(*dataset->Get(i));
    Local<Array> x = Array::Cast(*ex->Get(0));
    int k = 0;
    for(unsigned j = 0; j < x->Length(); ++j) {
      double xi = x->Get(j)->NumberValue();
      if (xi != 0){
        prob->x[i][k].index = j+1;
        prob->x[i][k].value = xi;
        k++;
      }
    }
  }
  delete[] nb_features;
  NanReturnUndefined();
}

// Handle<Value> ConvertDataToX(Local<Array> dataset, svm_node *x){
//   NanScope();
//   int counter = 0;
//   for (unsigned j=0; j < dataset->Length(); j++){
//     if (dataset->Get(j)->NumberValue() != 0){
//       counter++;
//     }
//   }

//   x = new svm_node[counter];
//   int k =0;
//   for (unsigned j=0; j < dataset->Length(); j++){
//     double xi = dataset->Get(j)->NumberValue();
//     if (xi != 0){
//       x[k].index = j+1;
//       x[k].value = xi;
//       k++;
//     }
//   }
//   NanReturnUndefined();
// }

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
    //svm_params->probability = params->Get(String::New("probability"))->IntegerValue();
  }

  obj->params = svm_params;
  const char *error_msg;
  error_msg =  svm_check_parameter(svm_params);
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
  
  struct svm_problem *prob = new svm_problem();
  getSvmProblemFromData(dataset, prob);
  
  svm_model *model = svm_train(prob, obj->params);
  svm_save_model("test.model", model);
  assert(model->l > 0);
  obj->model = model;
  // delete problem
  for (int i = 0; i< prob->l; i++) {
    delete[] prob->x[i];
  }
  delete[] prob->x;
  delete[] prob->y;
  delete prob;
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
  
  Local<Array> testset = Array::Cast(*args[0]->ToObject());
  // struct svm_problem *prob = new svm_problem(); 
  // //getSvmProblemFromData(testset, prob);
  
  // // TODO : Evaluate accuracy

  // // delete problem
  // for (int i = 0; i< prob->l; i++) {
  //   delete[] prob->x[i];
  // }
  // delete[] prob->x;
  // delete[] prob->y;
  // delete prob;
  NanReturnValue(Number::New(0));
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
  // svm_node *x;
  // ConvertDataToX(dataset, x);

  // double prediction = svm_predict(obj->model, x);

  // delete[] x;
  NanReturnValue(Number::New(0));
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
  // svm_node *x;
  // ConvertDataToX(dataset, x);

  // NanCallback *callback = new NanCallback(args[1].As<Function>());

  // double prediction = svm_predict(obj->model, x);
  // Local<Value> argv[] = {
  //   Number::New(prediction)
  // };
  // callback->Call(1, argv);

  // delete[] x;
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
  // svm_node *x;
  // ConvertDataToX(dataset, x);
  
  // double *prob_estimates = new double[obj->model->nr_class];
  // svm_predict_probability(obj->model,x,prob_estimates);
  // // Create a new empty array.
  Handle<Array> probs = Array::New(obj->model->nr_class);
  for (int j=0; j < obj->model->nr_class; j++){
    probs->Set(j, Number::New(j));
  }
  // delete[] prob_estimates;
  // delete[] x;
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