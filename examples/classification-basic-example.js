/**

 Simple example using C-SVC classificator (default) to predict the xor function

 Dataset : xor problem

 Note : because XOR dataset is to small, we set k-fold paramater to 1 to avoid cross validation
 **/
'use strict';
var so = require('stringify-object');
var nodesvm = require('../lib');

var xor = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];

// initialize predictor
var svm = new nodesvm.CSVC({
    kFold: 1
});

svm.train(xor)
    .spread(function (model, report) {
        console.log('SVM trained. \nReport :\n%s', so(report));

        console.log('Lets predict XOR values');
        xor.forEach(function(ex){
            var prediction = svm.predictSync(ex[0]);
            console.log('%d XOR %d => %d', ex[0][0], ex[0][1], prediction);
        });
    }).done(function () {
        console.log('done.');
    });

/* OUTPUT 
 SVM trained.
 Report :
 {
     "accuracy": 1,
     "fscore": 1,
     "recall": 1,
     "precision": 1,
     "class": {
         "0": {
             "precision": 1,
             "recall": 1,
             "fscore": 1
         },
         "1": {
             "precision": 1,
             "recall": 1,
             "fscore": 1
         }
     },
     "retainedVariance": 1
 }
 Lets predict XOR values
 0 XOR 0 => 0
 0 XOR 1 => 1
 1 XOR 0 => 1
 1 XOR 1 => 0
 done.
 */

