/**
  Perform ONE_CLASS evaluation as describe on
  http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#example-svm-plot-oneclass-py

 Training set is composed of 200 inliers examples (see `one-class.train.json`)
 Test set is composed of :
      - 100 inliers (labeled 1)
      - 20 outliers (labeled -1)
  (see `one-class.test.json`)

 Parameters:
   - nu = 0.1
   - RBF kernel with gamma = 0.1

 NOTE : No scaling / normalization used.
 Expected results:
    - accuracy : (120 - 4)/120 = 0.967%
    - precision on inliers: 100%
    - recall on outliers : 100%
*/
'use strict';

var so = require('stringify-object');
var Q = require('q');
var nodesvm = require('../lib');
var trainingFile = './examples/datasets/one-class.train.json';
var testingFile = './examples/datasets/one-class.test.json';

var svm = new nodesvm.OneClassSVM({
    nu: 0.1,
    kernelType: nodesvm.kernelTypes.RBF,
    gamma: 0.1,
    normalize: false,
    reduce: false,
    kFold: 1 // disable k-fold cross-validation
});

Q.all([
    nodesvm.read(trainingFile),
    nodesvm.read(testingFile)
]).spread(function (trainingSet, testingSet) {
    return svm.train(trainingSet)
        .spread(function (model, report) {
            return svm.evaluate(testingSet);
        });
}).done(function (evaluationReport) {
    console.log('Accuracy against the testset:\n', so(evaluationReport));
});


/*************************
 *        OUTPUT         *
 *************************

Accuracy against the testset:
 {
	accuracy: 0.9666666666666667,
	fscore: 0.9090909090909091,
	recall: 0.96,
	precision: 0.8333333333333334,
	class: {
		'1': {
			precision: 1,
			recall: 0.96,
			fscore: 0.9795918367346939,
			size: 100
		},
		'-1': {
			precision: 0.8333333333333334,
			recall: 1,
			fscore: 0.9090909090909091,
			size: 20
		}
	},
	size: 120
 }

*/
