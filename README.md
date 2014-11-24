# node-svm
[![Build Status](https://travis-ci.org/nicolaspanel/node-svm.png)](https://travis-ci.org/nicolaspanel/node-svm) [![Coverage Status](https://coveralls.io/repos/nicolaspanel/node-svm/badge.png?branch=master)](https://coveralls.io/r/nicolaspanel/node-svm?branch=master)


Support Vector Machine (SVM) library for nodejs.

[![NPM](https://nodei.co/npm/node-svm.png?downloads=true)](https://nodei.co/npm/node-svm/)

# Support Vector Machines
[Wikipedia](http://en.wikipedia.org/wiki/Support_vector_machine)  :

>Support vector machines are supervised learning models that analyze data and recognize patterns. 
>A special property is that they simultaneously minimize the empirical classification error and maximize the geometric margin; hence they are also known as maximum margin classifiers.
>[![Wikipedia image](http://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png)](http://en.wikipedia.org/wiki/File:Kernel_Machine.png)

# Installation
`npm install --save node-svm`

# How to use it
First of all, if you are not familiar with SVM, I highly recommend to read [this guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf).

Here's an example of using [node-svm](https://github.com/nicolaspanel/node-svm) to approximate the XOR function :
```javascript
var nodesvm = require('node-svm');
var xorProblem = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
];
var svm = new nodesvm.CSVC({ // classification 
  kernelType: nodesvm.KernelTypes.RBF,
  C: 1.0,
  gamma: 0.5
});

svm.once('trained', function(report) {
  // 'report' provides information about svm's accuracy
  [0,1].forEach(function(a){
    [0,1].forEach(function(b){
      var prediction = svm.predict([a, b]); 
      console.log("%d XOR %d => %d", a, b, prediction);
    });
  });
});

svm.train(xorProblem);

```
__Notes__ : 
 * There's no reason to use SVM to figure out XOR BTW...
 * The example show how to use `C-SVC` classifier but you can also use :
  * `NU-SVC` with `var svm = new nodesvm.NuSVC(options)` (classification)
  * `EPSILON-SVR` with `var svm = new nodesvm.EpsilonSVR(options)` (regression)
  * `NU-SVR` with `var svm = new nodesvm.NuSVR(options)` (regression)
 * `ONE-CLASS` SVM is not supported for now

More examples are available in the [examples folder](https://github.com/nicolaspanel/node-svm/tree/master/examples).

## Initialization
All possible options with default values are listed below : 
```javascript
var nodesvm = require('node-svm');

var svm = new nodesvm.SVM({
  svmType: nodesvm.SvmTypes.C_SVC,
  // kernels parameters
  kernelType: nodesvm.KernelTypes.RBF,  
  degree: [2,3,4],                      // for POLY kernel
  gamma: [0.03125, 0.125, 0.5, 2, 8],   // for POLY, RBF and SIGMOID kernels
  r: [0.125, 0.5, 2, 8],                // for POLY and SIGMOID kernels
  
  // SVM specific parameters
  C: [0.03125, 0.125, 0.5, 2, 8],       // cost for C_SVC, EPSILON_SVR and NU_SVR
  nu: [0.03125, 0.125, 0.5, 0.75, 1],   // for NU_SVC and NU_SVR
  epsilon: [0.03125, 0.125, 0.5, 2, 8], // for EPSILON-SVR

  // training options
  nFold: 4,               // for cross validation 
  normalize: true,        // whether to use mean normalization during data pre-processing
  reduce: true,           // whether to use PCA to reduce dataset dimension during data pre-processing
  retainedVariance: 0.99, // Define the acceptable impact on data integrity (if PCA activated)
  eps: 1e-3,              // stopping criteria 
  cacheSize: 100,         // cache siez in MB        
  probability : false     // whether to train a SVC or SVR model for probability estimates
});
```

__Notes__ : 
 * `degree`, `gamma`, `r`, `C`, `nu` and `epsilon` can take one or more values. For example  `degree: [2,3,4]` and `degree: 3` are both corrects
 * If at least one parameter has multiple values, `node-svm` will go through all the combinations to see which one gives the best predictions (i.e. it performs grid search to maximize [f-score](http://en.wikipedia.org/wiki/F1_score) for classification and minimize [Mean Squared Error](http://en.wikipedia.org/wiki/Mean_squared_error) for regression).

###Available kernels

 * Linear     : `nodesvm.KernelTypes.LINEAR`
 * Polynomial : `nodesvm.KernelTypes.POLY`
 * RBF        : `nodesvm.KernelTypes.RBF`
 * Sigmoid    : `nodesvm.KernelTypes.SIGMOID`

###Available SVM types

 * `C_SVC`      : multi-class classification
 * `NU_SVC`     : multi-class classification 
 * `EPSILON_SVR`: regression
 * `NU_SVR`     : regression

__Note__ : `ONE_CLASS` SVM is not supported (yet) 

##Training

SVMs can be trained using `svm#train(dataset, [callback])`

__Note__ :  Once trained, you can use `svm#getModel()` method to backup your svm model. Then you will be able to create new `svm` instances without having to train them again and again.

Pseudo code : 
```javascript
var svm = new nodesvm.SVM(options);

svm.train(dataset, function(){
  var model = svm.getModel();
  // persist your model...
});

on('something-append', function(){
 // get your model back...
 //...
 // create a new svm
 var newSvm = new nodesvm.SVM({model: model});
 // use it with no new training...
});
```

##Predictions
Once trained, you can use the `svm` object to predict values for given inputs. You can do that : 
 * Synchronously using `svm#predict(inputs)`
 * Asynchronously using `svm#predictAsync(inputs, callback)`

If you are working on a classification problem and **if you enabled probabilities during initialization** (see [initialization §](https://github.com/nicolaspanel/node-svm#initialization)), you can also predict probabilities for each class  : 
 * Synchronously using `svm#predictProbabilities(inputs)`. 
 * Asynchronously using `svm#predictProbabilitiesAsync(inputs, callback)`.

__Note__ : `inputs` must be a 1d array of numbers

## Features
node-svm provide additional features that allow you to :
 * [Mean normalize](http://en.wikipedia.org/wiki/Normalization_(statistics)) your dataset
 * Evaluate your `svm` against a test file
 * Perform cross validation on your dataset
 * Evaluate various combinaisons and find the best parameters (grid search)
 * Reduce your dataset dimension using [Principal Component Analysis (PCA)](http://en.wikipedia.org/wiki/Principal_component_analysis)

See [examples folder](https://github.com/nicolaspanel/node-svm/blob/master/examples) for more information.

# How it work

`node-svm` uses the official libsvm C++ library, version 3.18. For more information see also : 
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

A1 : Your dataset is empty or its format is incorrect.

A2 : Your dataset is too big.

###Difference between nu-SVC and C-SVC
Q : What is the difference between nu-SVC and C-SVC?

A : [Answer here](http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f411)

###Other questions
 * Take a look at [libsvm's FAQ](http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html).
 * Create [an issue](https://github.com/nicolaspanel/node-svm/issues)

# License
MIT

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/92d9dd8573d8b458d19a240629fea97a "githalytics.com")](http://githalytics.com/nicolaspanel/node-svm)
