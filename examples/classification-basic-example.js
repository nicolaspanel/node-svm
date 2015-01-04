/**

 Simple example using C-SVC classificator (default) to predict the xor function

 Dataset : xor problem

 Note : because XOR dataset is to small, we set k-fold paramater to 1 to avoid cross validation
 **/
'use strict';
var so = require('stringify-object');
var svm = require('../lib');

var xor = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];

// initialize predictor
var clf = new svm.CSVC({
    kFold: 1
});

clf.train(xor)
    .progress(function(progress){
        console.log('training progress: %d%', Math.round(progress*100));
    })
    .spread(function (model, report) {
        console.log('training report: %s\nPredictions:', so(report));
        xor.forEach(function(ex){
            var prediction = clf.predictSync(ex[0]);
            console.log('   %d XOR %d => %d', ex[0][0], ex[0][1], prediction);
        });
    });

/*************************
 *        OUTPUT         *
 *************************
 [training progress...]
 training report: {
	accuracy: 1,
	fscore: 1,
	recall: 1,
	precision: 1,
	class: {
		'0': {
			precision: 1,
			recall: 1,
			fscore: 1,
			size: 2
		},
		'1': {
			precision: 1,
			recall: 1,
			fscore: 1,
			size: 2
		}
	},
	size: 4,
	reduce: true,
	retainedVariance: 1,
	retainedDimension: 2,
	initialDimension: 2
}
 Predictions:
     0 XOR 0 => 0
     0 XOR 1 => 1
     1 XOR 0 => 1
     1 XOR 1 => 0
 */

