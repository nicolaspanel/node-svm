/** 
  Perform C-SVC classification as describe on the libsvm guide 
  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.

  training set : svmguide1.ds
  test set     : svmguide1.t.ds
  
  NOTE : No scaling / normalization used. Expect 66.925% accuracy with default parameters
*/
'use strict';

var Q = require('q'),
    nodesvm = require('../lib'),
    trainingFile = './examples/datasets/svmguide1.ds',
    testingFile = './examples/datasets/svmguide1.t.ds';

var svm = new nodesvm.CSVC({
    gamma: 0.25,
    c: 1, // allow you to evaluate several values during training
    normalize: false,
    reduce: false,
    kFold: 1 // disable k-fold cross-validation
});

Q.all([
    nodesvm.read(trainingFile),
    nodesvm.read(testingFile)
]).spread(function (trainingSet, testingSet) {
    return svm.train(trainingSet)
        .then(function () {
            return svm.evaluate(testingSet);
        });
}).done(function (evaluationReport) {
    console.log('Accuracy against the testset:\n', JSON.stringify(evaluationReport, null, '\t'));
});


/* OUTPUT 
SVM trained. Training report :
{
  "accuracy": 0.945919689119171,
  "fscore": 0.8900697016228053,
  "C": 1,
  "gamma": 0.25,
  "nbIterations": 2
}
Evaluation report against the testset:
{
  "nfold": 1,
  "accuracy": 0.66925,
  "fscore": 0.5079955373744887,
  "precision": 0.6022349743279976,
  "recall": 0.3415,
  "subsetsReports": [...]
}
*/