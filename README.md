# node-svm

Support Vector Machine (SVM) library for [nodejs](http://nodejs.org/) & [io.js](https://iojs.org/en/index.html) .

[![NPM](https://nodei.co/npm/node-svm.png)](https://nodei.co/npm/node-svm/)
[![Build Status](https://travis-ci.org/nicolaspanel/node-svm.png)](https://travis-ci.org/nicolaspanel/node-svm) [![Coverage Status](https://coveralls.io/repos/nicolaspanel/node-svm/badge.png?branch=master)](https://coveralls.io/r/nicolaspanel/node-svm?branch=master)

# Support Vector Machines
[Wikipedia](http://en.wikipedia.org/wiki/Support_vector_machine)  :

>Support vector machines are supervised learning models that analyze data and recognize patterns. 
>A special property is that they simultaneously minimize the empirical classification error and maximize the geometric margin; hence they are also known as maximum margin classifiers.
>[![Wikipedia image](http://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png)](http://en.wikipedia.org/wiki/File:Kernel_Machine.png)

# Installation
`npm install --save node-svm`

# Quick start
If you are not familiar with SVM I highly recommend this [guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf).

Here's an example of using [node-svm](https://github.com/nicolaspanel/node-svm) to approximate the XOR function :

```javascript
var svm = require('node-svm');

var xor = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];

// initialize a new predictor
var clf = new svm.CSVC();

clf.train(xor).done(function () {
    // predict things
    xor.forEach(function(ex){
        var prediction = clf.predictSync(ex[0]);
        console.log('%d XOR %d => %d', ex[0][0], ex[0][1], prediction);
    });
});

/******** CONSOLE ********
    0 XOR 0 => 0
    0 XOR 1 => 1
    1 XOR 0 => 1
    1 XOR 1 => 0
 */
```

More examples are available [here](https://github.com/nicolaspanel/node-svm/tree/master/examples).

__Note__: There's no reason to use SVM to figure out XOR BTW...


# API

## Classifiers

Possible classifiers are:

| Classifier  | Type                   | Params         | Initialization                |
|-------------|------------------------|----------------|-------------------------------|
| C_SVC       | multi-class classifier | `c`            | `= new svm.CSVC(opts)`        |
| NU_SVC      | multi-class classifier | `nu`           | `= new svm.NuSVC(opts)`       |
| ONE_CLASS   | one-class classifier   | `nu`           | `= new svm.OneClassSVM(opts)` |
| EPSILON_SVR | regression             | `c`, `epsilon` | `= new svm.EpsilonSVR(opts)`  |
| NU_SVR      | regression             | `c`, `nu`      | `= new svm.NuSVR(opts)`       |

## Kernels

Possible kernels are:

| Kernel  | Parameters                     |
|---------|--------------------------------|
| LINEAR  | No parameter                   |
| POLY    | `degree`, `gamma`, `r`         |
| RBF     |`gamma`                         |
| SIGMOID | `gamma`, `r`                   |


## Parameters and options

Possible parameters/options are:  

| Name             | Default value(s)       | Description                                                                                           |
|------------------|------------------------|-------------------------------------------------------------------------------------------------------|
| svmType          | `C_SVC`                | Used classifier                                                                                       | 
| kernelType       | `RBF`                  | Used kernel                                                                                           |
| c                | `[0.01,0.125,0.5,1,2]` | Cost for `C_SVC`, `EPSILON_SVR` and `NU_SVR`. Can be a `Number` or an `Array` of numbers              |
| nu               | `[0.01,0.125,0.5,1]`   | For `NU_SVC`, `ONE_CLASS` and `NU_SVR`. Can be a `Number` or an `Array` of numbers                    |
| epsilon          | `[0.01,0.125,0.5,1]`   | For `EPSILON_SVR`. Can be a `Number` or an `Array` of numbers                                         |
| degree           | `[2,3,4]`              | For `POLY` kernel. Can be a `Number` or an `Array` of numbers                                         |
| gamma            | `[0.001,0.01,0.5]`     | For `POLY`, `RBF` and `SIGMOID` kernels. Can be a `Number` or an `Array` of numbers                   |
| r                | `[0.125,0.5,0,1]`      | For `POLY` and `SIGMOID` kernels. Can be a `Number` or an `Array` of numbers                          |
| kFold            | `4`                    | `k` parameter for [k-fold cross validation]( http://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). `k` must be >= 1. If `k===1` then entire dataset is use for both testing and training.  |
| normalize        | `true`                 | Whether to use [mean normalization](http://en.wikipedia.org/wiki/Normalization_(statistics)) during data pre-processing  |
| reduce           | `true`                 | Whether to use [PCA](http://en.wikipedia.org/wiki/Principal_component_analysis) to reduce dataset's dimensions during data pre-processing  |
| retainedVariance | `0.99`                 | Define the acceptable impact on data integrity (require `reduce` to be `true`)                        |
| eps              | `1e-3`                 | Tolerance of termination criterion                                                                    |
| cacheSize        | `200`                  | Cache size in MB.                                                                                     |
| shrinking        | `true`                 | Whether to use the shrinking heuristics                                                               |
| probability      | `false`                | Whether to train a SVC or SVR model for probability estimates                                         |

The example below shows how to use them:

```javascript
var svm = require('node-svm');

var clf = new svm.SVM({
    svmType: 'C_SVC',
    c: [0.03125, 0.125, 0.5, 2, 8], 
    
    // kernels parameters
    kernelType: 'RBF',  
    gamma: [0.03125, 0.125, 0.5, 2, 8],
    
    // training options
    kFold: 4,               
    normalize: true,        
    reduce: true,           
    retainedVariance: 0.99, 
    eps: 1e-3,              
    cacheSize: 200,               
    shrinking : true,     
    probability : false     
});
```

__Notes__ :   
 * You can override default values by  creating a `.nodesvmrc` file (JSON) at the root of your project.
 * If at least one parameter has multiple values, [node-svm](https://github.com/nicolaspanel/node-svm/) will go through all possible combinations to see which one gives the best results (it performs grid-search to maximize [f-score](http://en.wikipedia.org/wiki/F1_score) for classification and minimize [Mean Squared Error](http://en.wikipedia.org/wiki/Mean_squared_error) for regression).


##Training

SVMs can be trained using `svm#train(dataset)` method.

Pseudo code : 
```javascript
var clf = new svm.SVM(options);

clf
.train(dataset)
.progress(function(rate){
    // ...
})
.spread(function(trainedModel, trainingReport){
    // ...
});
```

__Notes__ :  
 * `trainedModel` can be used to restore the predictor later (see [this example](https://github.com/nicolaspanel/node-svm/blob/master/examples/save-prediction-model-example.js) for more information).
 * `trainingReport` contains information about predictor's accuracy (such as MSE, precison, recall, fscore, retained variance etc.)

## Prediction
Once trained, you can use the classifier object to predict values for new inputs. You can do so : 
 * Synchronously using `clf#predictSync(inputs)`
 * Asynchronously using `clf#predict(inputs).then(function(predicted){ ... });`

**If you enabled probabilities during initialization**  you can also predict probabilities for each class  : 
 * Synchronously using `clf#predictProbabilitiesSync(inputs)`. 
 * Asynchronously using `clf#predictProbabilities(inputs).then(function(probabilities){ ... })`.

__Note__ : `inputs` must be a 1d array of numbers

## Model evaluation
Once the predictor is trained it can be evaluated against a test set. 

Pseudo code : 
```javascript
var svm = require('node-svm');
var clf = new svm.SVM(options);
 
svm.read(trainFile)
.then(function(dataset){
    return clf.train(dataset);
})
.then(function(trainedModel, trainingReport){
     return svm.read(testFile);
})
.then(function(testset){
    return clf.evaluate(testset);
})
.done(function(report){
    console.log(report);
});
 ```
# CLI

[node-svm](https://github.com/nicolaspanel/node-svm/) comes with a build-in Command Line Interpreter.

To use it you have to install [node-svm](https://github.com/nicolaspanel/node-svm/) globally using `npm install -g node-svm`.

See `$ node-svm -h` for complete command line reference.


## help
```shell
$ node-svm help [<command>]
```
Display help information about [node-svm](https://github.com/nicolaspanel/node-svm/) 


## train
```shell
$ node-svm train <dataset file> [<where to save the prediction model>] [<options>]
```
Train a new model with given data set

__Note__: use `$ node-svm train <dataset file> -i` to set parameters values dynamically.

## evaluate
```shell
$ node-svm evaluate <model file> <testset file> [<options>]
```
Evaluate model's accuracy against a test set

# How it work

`node-svm` uses the official libsvm C++ library, version 3.20. 

For more information see also : 
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
