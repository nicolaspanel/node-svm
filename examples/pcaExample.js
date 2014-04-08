/**
  Show how to use Principal Component Analysis (PCA) to reduce the datatset dimension
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing

**/

'use strict';

var nodesvm = require('../lib/nodesvm'),
    _ = require('underscore'),

    fileName = './examples/datasets/housing.ds';

nodesvm.readProblemAsync(fileName, function(housing){ 
  var norm = nodesvm.meanNormalizeDataSet(housing);

  var svm = new nodesvm.SVM({
    type: nodesvm.SvmTypes.EPSILON_SVR,
    kernel: new nodesvm.RadialBasisFunctionKernel(0.125),
    C: 8,
    epsilon: 0.125
  });

  var pca = nodesvm.reduceDatasetDimension(norm.dataset, 0.95);
  console.log('dataset dimensions reduced from %d to %d features', pca.oldDimension, pca.newDimension);
  console.log(pca.dataset);
  svm.train(pca.dataset); // svm is trained with normalized AND reduced dataset
  
  // test few examples from the original housing dataset
  var tests = _.sample(housing, 20);
  for (var i = 0; i < 20;  i++){
    var test = tests[i];
    var input = test[0];
    var normalizedInput = nodesvm.meanNormalizeInput(input, norm.mu, norm.sigma);
    var reducedInput = nodesvm.reduceInputDimension(normalizedInput, pca.U);
    console.log('{expected: %d, predicted: %d}', test[1], svm.predict(reducedInput));
  }
});