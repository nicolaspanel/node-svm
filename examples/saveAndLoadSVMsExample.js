/**
  Simple example using C-SVC classificator to demonstrate how to save and reuse SVM's model
  Dataset : xor.ds
**/

'use strict';

var nodesvm = require('../lib/nodesvm'),
    datasetFileName = './examples/datasets/xor.ds';

var svm = new nodesvm.CSVC({
  kernelType: nodesvm.KernelTypes.RBF,
  gamma: 0.5,
  C: 1
});

svm.once('trained', function  () {
	var newSvm = new nodesvm.SimpleSvm({model: svm.getModel()});

  [0,1].forEach(function(a){
    [0,1].forEach(function(b){
      console.log("%d XOR %d => %d", a, b, newSvm.predict([a, b]));
    });
  });
	process.exit(0);
});

svm.trainFromFile(datasetFileName);

/* OUTPUT 
0 XOR 0 => 0
0 XOR 1 => 1
1 XOR 0 => 1
1 XOR 1 => 0
*/