/**
  Simple example using C-SVC classificator to predict the xor function
  Dataset : xor.ds
  
  Note : 
   * we use 'nodesvm#loadSvmFromFile(path)' to create a new SVM from a file
   * To save an existing svm, make sure it is trained and use svm#saveToFile(path).
**/

'use strict';

var nodesvm = require('../lib/nodesvm'),
    datasetFileName = './examples/datasets/xor.ds',
    modelFileName = './examples/models/xor.model';

var svm = new nodesvm.CSVC({
  kernelType: nodesvm.KernelTypes.RBF,
  gamma: 0.5,
  C: 1
});

svm.once('trained', function  () {
  svm.saveToFile(modelFileName);

  var newSvm = new nodesvm.loadSvmFromFile(modelFileName);

  [0,1].forEach(function(a){
    [0,1].forEach(function(b){
      console.log("%d XOR %d => %d", a, b, svm.predict([a, b]));
    });
  }); 
});

svm.trainFromFile(datasetFileName);

/* OUTPUT 
0 XOR 0 => 0
0 XOR 1 => 1
1 XOR 0 => 1
1 XOR 1 => 0
*/