/**
  Simple example using EPSILON-SVR classificator to predict values
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing
  
  Note : libsvm#findBestParameters function help to find gamma, C and epsilon parameters
  that provie the lowest Mean Square Error () 
**/
'use strict';

var libsvm = require('../lib/nodesvm'),
    _ = require('underscore');

var nFold= 4,
    fileName = './examples/datasets/housing.ds';

libsvm.readAndNormalizeDatasetAsync(fileName, function(housing){ 
  console.log('Data set normalized with following parameters :');
  console.log('  * mu = \n', housing.mu);
  console.log('  * sigma = \n', housing.sigma);   

  console.log('Look for parameters that provide the lower Mean Square Error : ');
  var options = {
    svmType : libsvm.SvmTypes.EPSILON_SVR,
    kernelType : libsvm.KernelTypes.RBF,
    cValues: [0.03125, 0.125, 0.5, 2, 8],
    gValues: [8, 2, 0.5, 0.125, 0.03125],
    epsilonValues: [8, 2, 0.5, 0.125, 0.03125],
    log: true
  }; 
  libsvm.findBestParameters(housing.dataset, options, function(report) {
    // build SVM with found parameters
    var svm = new libsvm.SVM({
      type: libsvm.SvmTypes.EPSILON_SVR,
      kernel: new libsvm.RadialBasisFunctionKernel(report.gamma),
      C: report.C,
      epsilon: report.epsilon
    });
    var training = _.sample(housing.dataset, Math.round(housing.dataset.length * 0.8));
    var tests = _.sample(housing.dataset, 15);
    // train the svm
    svm.trainAsync(training, function(){
      // predict some values
      for (var i = 0; i < 20;  i++){
        var test = tests[i];
        console.log('{expected: %d, predicted: %d}', test[1], svm.predict(test[0]));
      }
    });
  }); 
});