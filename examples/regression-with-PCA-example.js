/**
  Show how to use Principal Component Analysis (PCA) to reduce the datatset's dimension
  
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing
  SVM TYPE : EPSILON-SVR (regression)

  NOTE : Also use mean normalizationfor better performences 
**/

'use strict';

var so = require('stringify-object');
var nodesvm = require('../lib');
var _a = require('mout/array');
var fileName = './examples/datasets/housing.ds';


var svm = new nodesvm.EpsilonSVR({
  gamma: [0.125, 0.5, 1],
  c: [8, 16, 32],
  epsilon: [0.001, 0.125, 0.5],
  normalize: true, // (default)
  reduce: true, // (default)
  retainedVariance: 0.995,
  kFold: 5
});


nodesvm.read(fileName)
    .then(function (dataset) {
        // train the svm with entire dataset
        return svm.train(dataset)
            .spread(function (model, report) {
                console.log('SVM trained. \nReport :\n%s', so(report));
                return dataset;
            });
    })
    .then(function (dataset) {
        // randomly pick m values and display predictions
        _a.pick(dataset, 5).forEach(function (ex, i) {
            var prediction = svm.predictSync(ex[0]);
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

 SVM trained.
 Report :
 {
	mse: 13.274198493484645,
	std: 3.640774430552408,
	mean: -0.13769545860496132,
	size: 506,
	retainedVariance: 0.995114672273733
 }
 { #1, expected: 24.4, predicted: 23.8999543384696}
 { #2, expected: 13.8, predicted: 13.300135350506823}
 { #3, expected: 16.2, predicted: 16.23335871065006}
 { #4, expected: 31.5, predicted: 32.19532581790853}
 { #5, expected: 13.9, predicted: 14.399888566573079}

done.

*/
