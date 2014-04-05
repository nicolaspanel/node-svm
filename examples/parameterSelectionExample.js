/** 
  Evaluate best parameter for C-SVC classification with RBF kernel

  training set : svmguide4.m.ds (concatenation of svmguide4.ds and svmguide4.t.ds)
*/
'use strict';

var libsvm = require('../lib/nodesvm'),
    _ = require('underscore');

var fileName = './examples/datasets/svmguide4.ds';

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
    console.log('Found parameters : ');
    console.log(" * Gamma = %d", report.gamma);
    console.log(" * C = %d", report.C);
    console.log(" * Accuracy = %d%%", report.accuracy * 100);
    console.log(" * F-Score = %d", report.fscore);
  });

});