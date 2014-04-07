/** 
  Perform C-SVC classification as describe on the libsvm guide 

  training set : svmguide1.ds
  test set     : svmguide1.t.ds

  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.
  NOTE : No scaling / normalization used. Expect accuracy with default params to be 66.925%
*/
'use strict';

var nodesvm = require('../lib/nodesvm'), 
    async = require('async');

var svc = new nodesvm.SVM({
  type: nodesvm.SvmTypes.C_SVC,
  kernel: new nodesvm.RadialBasisFunctionKernel(0.25),
  C: 1
});
var trainigSet = null,
    testset = null;

// Load problems
async.parallel([
  function (callback) {
    // load training set
    nodesvm.readProblemAsync('./examples/datasets/svmguide1.ds', function(dataset){
      trainigSet = dataset;
      // train svm
      console.log("Start training...");
      svc.trainAsync(trainigSet, function() {
        callback();
      });
      
    });
  }, 
  function (callback) {
    // load training set
    nodesvm.readProblemAsync('./examples/datasets/svmguide1.t.ds', function(dataset){
      testset = dataset;
      callback();
    });
  }
] , function(){
  // can start evaluation once svm trained and test set loaded
  svc.evaluate(testset, function(report){
    console.log('Evaluation result : \n', JSON.stringify(report, null, '\t')); 
  });
});

/* OUTPUT 
Start training...
Evaluation result : 
 {
  "nfold": 1,
  "accuracy": 0.66925,
  "fscore": 0.5079955373744887,
  "precision": 0.6022349743279976,
  "recall": 0.3415,
  "subsetsReports": [object]
}

*/