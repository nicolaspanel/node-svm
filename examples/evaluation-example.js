/** 
  Perform C-SVC classification as describe on the libsvm guide 
  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.

  training set : svmguide1.ds
  test set     : svmguide1.t.ds
  
  NOTE : No scaling / normalization used. Expect 66.925% accuracy with default parameters
*/
'use strict';

var so = require('stringify-object');
var Q = require('q');
var svm = require('../lib');
var trainingFile = './examples/datasets/svmguide1.ds';
var testingFile = './examples/datasets/svmguide1.t.ds';

var clf = new svm.CSVC({
    gamma: 0.25,
    c: 1, // allow you to evaluate several values during training
    normalize: false,
    reduce: false,
    kFold: 1 // disable k-fold cross-validation
});

Q.all([
    svm.read(trainingFile),
    svm.read(testingFile)
]).spread(function (trainingSet, testingSet) {
    return clf.train(trainingSet)
        .progress(function(progress){
            console.log('training progress: %d%', Math.round(progress*100));
        })
        .then(function () {
            return clf.evaluate(testingSet);
        });
}).done(function (evaluationReport) {
    console.log('Accuracy against the testset:\n', so(evaluationReport));
});


/*************************
 *        OUTPUT         *
 *************************

Accuracy against the testset:
 {
	accuracy: 0.66925,
	fscore: 0.5079955373744887,
	recall: 0.3415,
	precision: 0.6022349743279976,
	class: {
		'0': {
			precision: 0.9912917271407837,
			recall: 0.3415,
			fscore: 0.5079955373744887,
			size: 2000
		},
		'1': {
			precision: 0.6022349743279976,
			recall: 0.997,
			fscore: 0.7508943701751083,
			size: 2000
		}
	},
	size: 4000
}

*/
