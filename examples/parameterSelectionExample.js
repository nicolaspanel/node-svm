/** 
  Evaluate best parameter for C-SVC classification with RBF kernel
  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.10
  Training set : svmguide2.ds
  NODE : Normalization give a better performence than scaling (acc =97% instead of 85%)
*/
'use strict';

var libsvm = require('../lib/nodesvm'),
    _ = require('underscore');

var fileName = './examples/datasets/svmguide2.ds';

// Load problems
libsvm.readAndNormalizeDatasetAsync(fileName, function(svmguide){ 
  var cValues = _.map(_.range(-5, 15, 2), function(value){return Math.pow(2, value);}),
      gValues = _.map(_.range(3, -15, -2), function(value){return Math.pow(2, value);});

  var options = {
    svmType: libsvm.SvmTypes.C_SVC,
    kernelType: libsvm.KernelTypes.RBF,
    fold: 4,
    cValues:  cValues,
    gValues: gValues,
    log: true
  };
  console.log('Evaluation started (may take a minute)...');
  libsvm.findBestParameters(svmguide.dataset, options, function (report) {
    // body...
    console.log('done!'); 
    /* reports value are display in the console : 
    { 
      accuracy: 0.8,
      fscore: 0.6638888888888889,
      gamma: 0.001953125,
      C: 2048,
      nbIterations: 90 
    }
    */
  });

});