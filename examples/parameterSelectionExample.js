/** 
  Evaluate best parameter for C-SVC classification with RBF kernel
  
  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.10
  Training set : svmguide2.ds
  NODE : Normalization give a better performence than scaling (acc =97% instead of 85%)
*/
'use strict';

var nodesvm = require('../lib/nodesvm'),
    _ = require('underscore'),
    humanizeDuration = require("humanize-duration");

var fileName = './examples/datasets/svmguide2.ds';

// Load problems
nodesvm.readAndNormalizeDatasetAsync(fileName, function(svmguide){ 
  var options = {
    svmType: nodesvm.SvmTypes.C_SVC,    // (defaul option)
    kernelType: nodesvm.KernelTypes.RBF,// (defaul option)
    fold: 4,
    cValues:  _.map(_.range(-5, 15, 2), function(v){return Math.pow(2, v);}),
    gValues: _.map(_.range(3, -15, -2), function(v){return Math.pow(2, v);}),
  };
  console.log('Evaluation started (may take a minute)...');
  nodesvm.findBestParameters(svmguide.dataset, options, function (report) {

    console.log('Evaluation restult : \n', JSON.stringify(report, null, '\t')); 
    
  }, function(progressRate, remainingTime){
    // called during evaluation to report progress
    // remainingTime in ms
    if ((progressRate*100)%10 === 0){
      console.log('%d% - %s remaining...', progressRate * 100, humanizeDuration(remainingTime));
    }
  });

});

/* OUTPUT
Evaluation started (may take a minute)...
10% - 34 seconds, 398 milliseconds remaining...
20% - 24 seconds, 56 milliseconds remaining...
30% - 18 seconds, 347 milliseconds remaining...
40% - 13 seconds, 680 milliseconds remaining...
50% - 10 seconds, 301 milliseconds remaining...
60% - 7 seconds, 513 milliseconds remaining...
70% - 5 seconds, 238 milliseconds remaining...
80% - 3 seconds, 365 milliseconds remaining...
90% - 1 second, 630 milliseconds remaining...
100% - 0 remaining...
Evaluation restult : 
 {
  "accuracy": 0.9484536082474226,
  "fscore": 0.9265644955300127,
  "C": 8,
  "gamma": 0.03125,
  "nbIterations": 90
}
*/