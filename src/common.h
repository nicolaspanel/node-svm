#ifndef _LIBSVM_COMMON_H
#define _LIBSVM_COMMON_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <node.h>
#include <assert.h>
#include "../node_modules/nan/nan.h"
#include "./libsvm-317/svm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace v8;

inline Handle<Value> LIBSVM_Exception(const char *msg){
  HandleScope scope;
  return ThrowException(Exception::Error(String::New(msg)));
}

/**
 * Reads a value at the given key and returns a C boolean value
 * @param o: the JS object on which to read at the given key
 * @param name: the key to read from
 * @return the boolean value read from the object
 */
inline bool LIBSVM_BOOL_KEY(Handle<Object> o, const char* name) {
  assert(o->IsObject());
  Handle<Value> value = o->Get(String::New(name));
  assert(value->IsBoolean());
  return value->BooleanValue();
}

/**
 * Reads a value at the given key and returns a C int value
 * @param o: the JS object on which to read at the given key
 * @param name: the key to read from
 * @return the int value read from the object
 */
inline int LIBSVM_INT_KEY(Handle<Object> o, const char* name) {
  assert(o->IsObject());
  Handle<Value> value = o->Get(String::New(name));
  assert(value->IsNumber());
  assert(value->IsUint32());
  return value->Int32Value();
}

/**
 * Reads a value at the given key and returns a C string value
 * NOTE: this function allocates the needed space for the string
 * it is the responsibility of the caller to free this pointer
 * @param o: the JS object on which to read at the given key
 * @param name: the key to read from
 * @return the string value read from the object
 */
inline char* LIBSVM_STRING_KEY(Handle<Object> o, const char* name) {
  assert(o->IsObject());
  Handle<Value> value = o->Get(String::New(name));
  if(value->IsNull()) {
    return NULL;
  }
  assert(value->IsString());

  char* v = new char[value->ToString()->Length()+1];
  strcpy(v, *(String::AsciiValue(value)));
  return v;
}

/**
 * namespace in which I declare most of the stuff I need
 */
namespace libsvm {
  inline struct svm_problem convert_data_to_problem(Local<Array> data){
    
    unsigned nb_examples = data->Length();
    std::cout << "Problem contains" << nb_examples << "examples" << std::endl;
    struct svm_problem prob;
    prob.l = nb_examples;
    if (prob.l == 0)
      return prob;
    int elements = 0;
    for (unsigned i=0; i < nb_examples; i++) {
      Local<Object> ex = data->Get(i)->ToObject();
      Local<Array> x = Array::Cast(*ex->Get(String::New("x"))->ToObject());
      for (unsigned j=0; j < x->Length(); j++){
        elements++;
      }
    }

    prob.y = Malloc(double,nb_examples);
    prob.x = Malloc(struct svm_node *,nb_examples);
    struct svm_node *x_space = Malloc(struct svm_node,elements);
    int k =0;
    for (unsigned i=0; i < nb_examples; i++) {
      Local<Object> ex = data->Get(i)->ToObject();
      prob.y[i] = ex->Get(String::New("y"))->NumberValue();
      
      Local<Array> x = Array::Cast(*ex->Get(String::New("x"))->ToObject());
      prob.x[i] = &x_space[k];
      for (unsigned j=0; j < x->Length(); j++){
        x_space[k].index = j+1;
        x_space[k].value = x->Get(j)->NumberValue();
        k++;
      }
    }
    return prob;
  }
}

#endif /* _LIBSVM_COMMON_H */