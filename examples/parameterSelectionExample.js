/** 
  Evaluate best parameter for C-SVC classification with RBF kernel

  Training set : svmguide2.ds
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
    gValues: gValues
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