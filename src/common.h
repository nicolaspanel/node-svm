#ifndef _LIBSVM_COMMON_H
#define _LIBSVM_COMMON_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <node.h>
#include <assert.h>
#include "../node_modules/nan/nan.h"
#include "./libsvm-317/svm.h"


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

#endif /* _LIBSVM_COMMON_H */