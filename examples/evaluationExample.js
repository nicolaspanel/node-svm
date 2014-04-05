/** 
  Perform C_SVC classification as describe on the libsvm guide 

  training set : svmguide1.ds
  test set     : svmguide1.t.ds

  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.
  NOTE : No scaling / normalization used. Expecterd accuracy with default params to be 66.925%
*/
'use strict';

var libsvm = require('../lib/nodesvm'), 
    async = require('async');

var cSVM = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(0.25),
  C: 1
});
var trainigSet = null,
    testset = null;

// Load problems
async.parallel([
  function (callback) {
    // load training set
    libsvm.readProblemAsync('./examples/datasets/svmguide1.ds', function(dataset){
      trainigSet = dataset;
      // train svm
      console.log("Start training...");
      cSVM.trainAsync(trainigSet, function() {
        callback();
      });
      
    });
  }, 
  function (callback) {
    // load training set
    libsvm.readProblemAsync('./examples/datasets/svmguide1.t.ds', function(dataset){
      testset = dataset;
      callback();
    });
  }
] , function(){
  // can start evaluation once svm trained and test set loaded
  console.log("Evaluation Report:");
  cSVM.evaluate(testset, function(report){
    console.log(" * Accuracy = %d%%", report.accuracy * 100);
    console.log(" * F-Score = %d", report.fscore);
    console.log(" * Precision = %d", report.precision);
    console.log(" * Recall = %d", report.recall);
  });
});
