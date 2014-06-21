/**
  Show how to use Principal Component Analysis (PCA) to reduce the datatset's dimension
  
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing
  SVM TYPE : EPSILON-SVR (regression)

  NOTE : Also use mean normalizationfor better performences 
**/

'use strict';

var nodesvm = require('../lib/nodesvm'),
    _ = require('underscore'),
    fileName = './examples/datasets/housing.ds';

var testsamples = null;

var svm = new nodesvm.EpsilonSVR({
  kernelType: nodesvm.KernelTypes.RBF,
  gamma: [0.125, 0.5, 2],
  C: [8, 16, 32],
  epsilon: [0.001, 0.125, 0.5],
  normalize: true, // default value
  reduce: true, // default value
  retainedVariance: 0.98
});

svm.once('dataset-reduced', function(oldDim, newDim, retainedVar){
  console.log('Dataset reduced from %d to %d features using PCA.', oldDim, newDim);
  console.log('%d% of the variance have been retained.\n', retainedVar * 100);
});

svm.once('trained', function(report){
  console.log('SVM trained. Report :\n%s', JSON.stringify(report, null, '\t'));
  console.log('SVM trained. Lets predict some values : ');
  for (var i = 0; i < testsamples.length;  i++){
    var test = testsamples[i];
    var inputs = test[0];
    var expected = test[1];
    var prediction = svm.predict(inputs);
    console.log(' { #%d, expected: %d, predicted: %d}',i+1, expected, prediction);
  }
	process.exit(0);
});

nodesvm.readDatasetAsync(fileName, function (ds) {
  ds = _.shuffle(ds);
  var trainingsetSize = Math.round(0.95 * ds.length);
  var traininset = _.first(ds, trainingsetSize);
  var testset = _.last(ds, ds.length - trainingsetSize);
  testsamples = _.sample(testset, 20);
  svm.train(traininset);
});

/* OUTPUT
Dataset reduced from 13 to 11 features using PCA.
98.20% of the variance have been retained.

SVM trained. Report :
{
  "mse": 5.605383189519437,
  "C": 32,
  "gamma": 0.5,
  "epsilon": 0.125,
  "nbIterations": 27
}
SVM trained. Lets predict some values : 
 { #1, expected: 33.1, predicted: 31.872021937864574}
 { #2, expected: 7.2, predicted: 7.759332979168485}
 { #3, expected: 22.4, predicted: 24.661293282902776}
 { #4, expected: 16.7, predicted: 17.23850569448206}
 { #5, expected: 23.2, predicted: 22.59888436738391}
 { #6, expected: 25.2, predicted: 27.66896902786514}
 { #7, expected: 24.1, predicted: 21.185879126591416}
 { #8, expected: 32.5, predicted: 25.944815405122178}
 { #9, expected: 19.9, predicted: 20.07654893587641}
 { #10, expected: 24.4, predicted: 23.607902550407395}
 { #11, expected: 19.9, predicted: 17.008426112160603}
 { #12, expected: 23.1, predicted: 22.88491868303525}
 { #13, expected: 20.3, predicted: 21.33280841773401}
 { #14, expected: 19.4, predicted: 19.83713955583848}
 { #15, expected: 23.8, predicted: 23.413447472664327}
 { #16, expected: 34.9, predicted: 37.65820946495587}
 { #17, expected: 14.9, predicted: 12.816421129579158}
 { #18, expected: 16.2, predicted: 15.959022651206888}
 { #19, expected: 27.1, predicted: 26.396166945355507}
 { #20, expected: 22, predicted: 22.883406497245783}
*/