node-svm
========

[libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) (Support Vector Machine library) addon for nodejs

[![Build Status](https://travis-ci.org/nicolaspanel/node-svm.png)](https://travis-ci.org/nicolaspanel/node-svm)
[![Coverage Status](https://coveralls.io/repos/nicolaspanel/node-svm/badge.png?branch=master)](https://coveralls.io/r/nicolaspanel/node-svm?branch=master)

[![NPM](https://nodei.co/npm/node-svm.png?downloads=true)](https://nodei.co/npm/node-svm/)

# How to use it
First of all, if you are not familiar with SVM, I highly recommend to read [this guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf).

Here's an example of using it to approximate the XOR function :
```javascript
var nodesvm = require('node-svm');
var xorProblem = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
];
var svm = new nodesvm.SVM({
  type: nodesvm.SvmTypes.C_SVC,
  kernel: new nodesvm.RadialBasisFunctionKernel(0.5),
  C: 1.0
});

svm.train(xorProblem);

xorProblem.forEach(function(ex){
  svm.predict(ex[0]).should.equal(ex[1]);
});

```
Notice : There's no reason to use SVM to figure out XOR BTW...

More examples are available in the [same name folder](https://github.com/nicolaspanel/node-svm/tree/master/examples).

## Initialisation
Initialization arguments with default values are listed below : 
```javascript
var nodesvm = require('node-svm');

var svm = new nodesvm.SVM({
  type: nodesvm.SvmTypes.C_SVC,  // see supported types below
  kernel: new nodesvm.RadialBasisFunctionKernel(gamma), // see other kernels below
  C: 0.1,  // Cost parameter. Required for C_SVC, EPSILON_SVR and NU_SVR. Must be greater than zero
  nu: 0.5, // nu parameter. Required for NU_SVC, ONE_CLASS and NU_SVR. Must be within 0 and 1
  epsilon : 0.1, // Epsilon parameter, required for epsilon-SVR. Must me greater than zero
  
  // training options
  eps: 1e-3, // stopping criteria 
  cacheSize: 100, // memory size in MB  
  probability : false // whether to train a SVC or SVR model for probability estimates (has a significant impact on the duration of the training)
});
```
Notice : `nodesvm#SVM` function will throw an exception if you provide incorrect parameters.

###Available kernels

 * Linear     : `var kernel = new nodesvm.LinearKernel();`
 * Polynomial : `var kernel = new nodesvm.PolynomialKernel(degree, gamma, r);`
 * RBF        : `var kernel = new nodesvm.RadialBasisFunctionKernel(gamma);`
 * Sigmoid    : `var kernel = new nodesvm.SigmoidKernel(gamma, r);`

Default parameters values : 
 * gamma: 2.0
 * degree: 3.0
 * r: 0.0

###Available SVM types

 * `C_SVC`      : multi-class classification
 * `NU_SVC`     : multi-class classification
 * `ONE_CLASS`  : one-class SVM  
 * `EPSILON_SVR`: regression
 * `NU_SVR`     : regression

##Training
SVMs can be trained : 
 * Synchronously using `svm#train()` method
 * Asynchronously using `svm#trainAsync(callback)` method

Notice :  Once trained, you can use `svm#saveToFile(path)` method to backup your svm model. Then you will be able to create new `svm` instances without having to train them again and again.

Pseudo code : 
```javascript
var svm = new nodesvm.SVM(options);
svm.train(); // svm need to be trained before you can save it
svm.saveToFile('./path/to/myFile.model');
//...
var svm2 = new nodesvm.SVM({file: './path/to/myFile.model'});
svm2.predict(values);
// ...
```

##Predictions
Once trained, you can use your `svm` to predict values for given inputs. As before, you can do that : 
 * Synchronously using `svm#predict(inputs)` method. 
 * Asynchronously using `svm#predictAsync(inputs, callback)` method.

Notice : `inputs` must be an array of numbers

## Features
node-svm provide additional features that allow you to :
 * [Mean normalize](http://en.wikipedia.org/wiki/Normalization_(statistics)) your dataset. See [classification example](https://github.com/nicolaspanel/node-svm/blob/master/examples/classificationBasicExample.js)
 * Evaluate your `svm` against a test file. See [evaluation example](https://github.com/nicolaspanel/node-svm/blob/master/examples/evaluationExample.js)
 * Perform cross validation on your dataset. See [cross validation example](https://github.com/nicolaspanel/node-svm/blob/master/examples/crossValidationExample.js)
 * Evaluate various combinaisons and find the best parameters. See [evaluation example](https://github.com/nicolaspanel/node-svm/blob/master/examples/parameterSelectionExample.js)

# How it work
`node-svm` uses the official libsvm C++ library, version 3.18. For more informations, see also : 
 * [libsvm web site](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
 * Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.
 * [Wikipedia article about SVM](https://en.wikipedia.org/wiki/Support_vector_machine)
 * [node addons](http://nodejs.org/api/addons.html)

# Contributions
Feel free to fork and improve/enhance `node-svm` in any way your want.

If you feel that the community will benefit from your changes, please send a pull request : 
 * Fork the project.
 * Make your feature addition or bug fix.
 * Add documentation if necessary.
 * Add tests for it. This is important so I don't break it in a future version unintentionally (run `grunt` or `npm test`).
 * Send a pull request to the `develop` branch. 

#FAQ
###Segmentation fault
Q : Node returns 'segmentation fault' error during training. What's going on?

A : Your dataset is empty or its format is incorrect.

###Difference between nu-SVC and C-SVC
Q : What is the difference between nu-SVC and C-SVC?

A : [Answer here](http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f411)

###Other questions
 * Take a look at [libsvm's FAQ](http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html).
 * Create [an issue](https://github.com/nicolaspanel/node-svm/issues)

# License
MIT

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/92d9dd8573d8b458d19a240629fea97a "githalytics.com")](http://githalytics.com/nicolaspanel/node-svm)
