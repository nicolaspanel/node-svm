node-svm
========

`node-svm` is a wrapper of [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) (Support Vector Machine library) for nodejs.

**Status** : under development (0.1.5).
[![Build Status](https://travis-ci.org/nicolaspanel/node-svm.png)](https://travis-ci.org/nicolaspanel/node-svm)

# Installation

`npm install node-svm --save`

# How to use it
First of all, if you are not familiar with SVM, I highly recommend to read [this guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf).

Here's an example of using it to approximate the XOR function
```javascript
var libsvm = require('node-svm');
var xorProblem = [
  { x: [-1, -1], y: -1 },
  { x: [-1,  1], y:  1 },
  { x: [ 1, -1], y:  1 },
  { x: [ 1,  1], y: -1 }
];
var svm = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(gamma),
  C: C
});
xorProblem.forEach(function(ex){
  svm.predict(ex.x).should.equal(ex.y);        // !!! NOT always true
  [-1,1].should.containEql(svm.predict(ex.x));  // always true
});

```
BTW, there's no reason to use SVM to figure out XOR...


## Arguments and options
Options with default values are listed below : 
```javascript
var args = {
  //required
  type        : ... // see below 
  kernel      : ... // see below
  C           : ... // required for C_SVC, epsilon_VR, and nu-SVR
  
  // options
  cacheSize   : 100,  //MB
  eps         : 1e-3, // tolerance of termination criterion 
  shrinking   : true, // whether to use the shrinking heuristics
  probability : true  //  do probability estimates
};
var svm = new libsvm.SVM(args);
```
Available kernels are  : 
 * Linear     : `var kernel = new libsvm.LinearKernel();`
 * Polynomial : `var kernel = new libsvm.PolynomialKernel(degree, gamma, r);`
 * RBF        : `var kernel = new libsvm.RadialBasisFunctionKernel(gamma);`
 * Sigmoid    : `var kernel = new libsvm.SigmoidKernel(gamma, r);`

Available SVM are : 
 * `C_SVC`      : multi-class classification
 * `NU_SVC`     : multi-class classification
 * `ONE_CLASS`  : one-class SVM  
 * `EPSILON_SVR`: regression
 * `NU_SVR`     : regression

Note : See difference between nu-SVC and C-SVC [here](http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f411).  

# How it work
`node-svm` uses the official libsvm C++ library, version 3.17. It also provides helpers to facilitate its use in real projects.

For more informations, see also : 
 * [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
 * [Wikipedia article about SVM](https://en.wikipedia.org/wiki/Support_vector_machine)
 * [node addons](http://nodejs.org/api/addons.html)

# Contributions
Feel free to fork and improve/enhance `node-svm` in any way your want.

If you feel that the community will benefit from your changes, please send a pull request following steps below : 
 * Fork the project.
 * Make your feature addition or bug fix.
 * Add documentation if necessary.
 * Add tests for it. This is important so I don't break it in a future version unintentionally (run `grunt` or `npm test`).
 * Send a pull request. 

# License
MIT