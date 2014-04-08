/**
  Show how to use Principal Component Analysis (PCA) to reduce the datatset dimension
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing

**/

'use strict';

var nodesvm = require('../lib/nodesvm'),
    _ = require('underscore'),

    fileName = './examples/datasets/housing.ds';

nodesvm.readDatasetAsync(fileName, function(housing){ 
  var norm = nodesvm.meanNormalizeDataSet(housing);

  var svm = new nodesvm.SVM({
    type: nodesvm.SvmTypes.EPSILON_SVR,
    kernel: new nodesvm.RadialBasisFunctionKernel(0.125),
    C: 8,
    epsilon: 0.125
  });

  var pca = nodesvm.reduceDatasetDimension(norm.dataset, 0.95);
  console.log('Dataset dimensions reduced from %d to %d', pca.oldDimension, pca.newDimension);
  console.log('%d% of the variance retained', pca.retainedVariance);
  svm.trainAsync(pca.dataset, function(){ // svm is trained with normalized AND reduced dataset
    console.log('SVM trained. Lets predict some values : ');
    // test few examples from the original housing dataset
    var tests = _.sample(housing, 20);
    for (var i = 0; i < 20;  i++){
      var test = tests[i];
      var input = test[0];
      var normalizedInput = nodesvm.meanNormalizeInput(input, norm.mu, norm.sigma);
      var reducedInput = nodesvm.reduceInputDimension(normalizedInput, pca.U);
      console.log(' { #%d, expected: %d, predicted: %d}',i+1, test[1], svm.predict(reducedInput));
    }
  }); 
  
});

/* OUTPUT
Dataset dimensions reduced from 13 to 9
0.95084...% of the variance retained
SVM trained. Lets predict some values : 
 { #1, expected: 21, predicted: 21.07332172526786}
 { #2, expected: 17.2, predicted: 11.149460046430221}
 { #3, expected: 20.3, predicted: 21.014438097028318}
 { #4, expected: 18.7, predicted: 19.643511125864002}
 { #5, expected: 11.3, predicted: 11.174765918482318}
 { #6, expected: 23.3, predicted: 21.883181622717363}
 { #7, expected: 19.4, predicted: 19.407808736763364}
 { #8, expected: 10.5, predicted: 9.232996295870505}
 { #9, expected: 23.1, predicted: 22.15947089252767}
 { #10, expected: 50, predicted: 42.25849498458795}
 { #11, expected: 14.6, predicted: 14.724619219541664}
 { #12, expected: 34.9, predicted: 31.37582769444906}
 { #13, expected: 21, predicted: 20.753157299409438}
 { #14, expected: 20, predicted: 20.428756008139302}
 { #15, expected: 20.6, predicted: 20.497810778090564}
 { #16, expected: 19.9, predicted: 18.28620207782037}
 { #17, expected: 13.1, predicted: 14.7286042701368}
 { #18, expected: 29.1, predicted: 28.975202207688525}
 { #19, expected: 21.2, predicted: 21.8036303845415}
 { #20, expected: 19.4, predicted: 18.001545186784767}
*/