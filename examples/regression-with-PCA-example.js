/**
  Show how to use Principal Component Analysis (PCA) to reduce the datatset's dimension
  
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing
  SVM TYPE : EPSILON_SVR (regression)

  NOTE : Also use mean normalization for better performance
**/

'use strict';

var so = require('stringify-object');
var svm = require('../lib');
var _a = require('mout/array');
var fileName = './examples/datasets/housing.ds';


var clf = new svm.EpsilonSVR({
  gamma: [0.125, 0.5, 1],
  c: [8, 16, 32],
  epsilon: [0.001, 0.125, 0.5],
  normalize: true, // (default)
  reduce: true, // (default)
  retainedVariance: 0.995,
  kFold: 5
});


svm.read(fileName)
    .then(function (dataset) {
        // train the svm with entire dataset
        return clf.train(dataset)
            .progress(function(progress){
                console.log('training progress: %d%', Math.round(progress*100));
            })
            .spread(function (model, report) {
                console.log('SVM trained. \nReport :\n%s', so(report));
                return dataset;
            });
    })
    .then(function (dataset) {
        // randomly pick m values and display predictions
        _a.pick(dataset, 5).forEach(function (ex, i) {
            var prediction = clf.predictSync(ex[0]);
            console.log(' { #%d, expected: %d, predicted: %d}',i+1, ex[1], prediction);
        });
    })
    .fail(function (err) {
        throw err;
    })
    .done(function () {
        console.log('done.');
    });


/*************************
 *        OUTPUT         *
 *************************

 [training progress... ]
 SVM trained.
 Report :
 {
     mse: 12.840723874926569,
     std: 3.5807999959488956,
     mean: -0.13636445262226565,
     size: 506,
     reduce: true,
     retainedVariance: 0.995114672273733,
     retainedDimension: 12,
     initialDimension: 13
 }
 { #1, expected: 28, predicted: 28.49960798991779}
 { #2, expected: 25, predicted: 24.499820075688632}
 { #3, expected: 23, predicted: 22.70164253213565}
 { #4, expected: 26, predicted: 26.174256533573505}
 { #5, expected: 18, predicted: 17.488069908365038}
 done.

*/
