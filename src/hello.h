#ifndef _HELLO_H
#define _HELLO_H

using namespace v8;

// Simple synchronous access to the `Estimate()` function
NAN_METHOD(Hello) {
  NanScope();
  NanReturnValue(String::New("World!"));
}

#endif /* _HELLO_H */