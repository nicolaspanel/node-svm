/**
  Show how to use Principal Component Analysis (PCA) to reduce the datatset's dimension
  
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing
  SVM TYPE : EPSILON-SVR (regression)

  NOTE : Also use mean normalizationfor better performences 
**/

'use strict';

var nodesvm = require('../lib'),
    _a = require('mout/array'),
    fileName = './examples/datasets/housing.ds';


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
                console.log('SVM trained. \nReport :\n%s', JSON.stringify(report, null, '\t'));
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
        console.log('done');
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