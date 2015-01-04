/**
 Simple example using C-SVC classificator to demonstrate how to save and reuse SVM's model
 Dataset : xor.ds
 **/

'use strict';

var svm = require('../lib');

var xor = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];

var clf = new svm.CSVC();

clf.train(xor).spread(function (model, report) {
    var newClf = svm.restore(model);
    console.log('Lets predict XOR with new Classifier:');
    xor.forEach(function (ex) {
        var prediction = newClf.predictSync(ex[0]);
        console.log('   %d XOR %d => %d', ex[0][0], ex[0][1], prediction);
    });
});

/*************************
 *        OUTPUT         *
 *************************

 Lets predict XOR with new Classifier:
     0 XOR 0 => 0
     0 XOR 1 => 1
     1 XOR 0 => 1
     1 XOR 1 => 0

 */

