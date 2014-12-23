/**

 Simple example using C-SVC classificator (default) to predict the xor function

 Dataset : xor problem

 Note : because XOR dataset is to small, we set k-fold paramater to 1 to avoid cross validation
 **/
'use strict';
var svm = require('../lib');

var xor = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];

// initialize predictor
var clf = new svm.CSVC();

clf.train(xor)
    .spread(function (model, report) {
        xor.forEach(function(ex){
            var prediction = clf.predictSync(ex[0]);
            console.log('%d XOR %d => %d', ex[0][0], ex[0][1], prediction);
        });
    });

/*************************
 *        OUTPUT         *
 *************************
 0 XOR 0 => 0
 0 XOR 1 => 1
 1 XOR 0 => 1
 1 XOR 1 => 0
 */

