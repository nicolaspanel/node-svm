/** 
  Perform C-SVC classification as describe on the libsvm guide 
  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.

  training set : svmguide1.ds
  test set     : svmguide1.t.ds
  
  NOTE : No scaling / normalization used. Expect accuracy with default params to be 66.925%
*/
'use strict';

var nodesvm = require('../lib/nodesvm'), 
    trainingFile = './examples/datasets/svmguide1.ds',
    testingFile = './examples/datasets/svmguide1.t.ds';

var svm = new nodesvm.CSVC({
  kernelType: nodesvm.KernelTypes.RBF,
  gamma: 0.25,
  C: [0.5, 1], // allow you to evaluate several values during training
  normalize: false,
  reduce: false
});

svm.once('trained', function(report) {
  console.log('SVM trained. Training report :\n%s', JSON.stringify(report, null, '\t'));
  
  nodesvm.readDatasetAsync(testingFile, function(testset){
	  svm.evaluate(testset, function(evalReport){
      console.log('Evaluation report against the testset:\n%s', JSON.stringify(evalReport, null, '\t'));
	    process.exit(0);
    });
  });
});

svm.trainFromFile(trainingFile);

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