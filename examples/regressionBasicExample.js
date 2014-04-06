/**
  Simple example using C-SVC classificator to predict the xor function
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing
  
  Note : because dataset is normalized we also normalize inputs
**/
'use strict';

var libsvm = require('../lib/nodesvm');
var fileName = './examples/datasets/housing.ds';
var svm = new libsvm.SVM({
  type: libsvm.SvmTypes.EPSILON_SVR,
  kernel: new libsvm.RadialBasisFunctionKernel(0.5),
  C: 1,
  epsilon: 0.1
});

libsvm.readAndNormalizeDatasetAsync(fileName, function(housing){ 
  console.log('Data set normalized with following parameters :');
  console.log('  * mu = \n', housing.mu);
  console.log('  * sigma = \n', housing.sigma); 
  svm.trainAsync(housing.dataset, function() {
    console.log("trained");
  });
});