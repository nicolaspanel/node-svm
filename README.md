node-svm
========

`node-svm` is a wrapper of [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) (Support Vector Machine library) for nodejs.

**Status** : alpha version (0.1.0).

# Installation

`npm install node-svm --save`

# How to use it

libsvm provide both classification and regression predictors.

## Classification example
Here's an example of using it to approximate the XOR function
```javascript
var libsvm = require('../lib/node-svm');
var xorProblem = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 }
];
var svm = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(gamma),
  C: C
});
xorProblem.forEach(function(ex){
  svm.predict(ex.x).should.equal(ex.y);        // !not always true
  [0,1].should.containEql(svm.predict(ex.x));  // always true
});

```
BTW, there's no reason to use SVM to figure out XOR...

## Regression example
```javascript
```

## Options
Options with default values are listed below : 
```javascript
var options = {
  cacheSize   : 100,  //MB
  eps         : 1e-3, // epsilon 
  shrinking   : true, // use the shrinking heuristics
  probability : true  //  do probability estimates
};
```
