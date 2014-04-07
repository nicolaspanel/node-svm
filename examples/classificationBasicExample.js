/**
  Simple example using C-SVC classificator to predict the xor function
  Dataset : xor problem loaded from file (libsvm format)
  
  Note : because dataset is normalized we also normalize inputs
**/
'use strict';

var nodesvm = require('../lib/nodesvm');
var fileName = './examples/datasets/xor.ds';
var svm = new nodesvm.SVM({
  type: nodesvm.SvmTypes.C_SVC,
  kernel: new nodesvm.RadialBasisFunctionKernel(0.5),
  C: 1
});

nodesvm.readAndNormalizeDatasetAsync(fileName, function(xor){ 
  var mu = xor.mu,
      sigma = xor.sigma;
  
  svm.trainAsync(xor.dataset, function() {
    [0,1].forEach(function(a){
      [0,1].forEach(function(b){
        var normalizedInput = nodesvm.meanNormalizeInput([a, b], mu, sigma);
        var prediction = svm.predict(normalizedInput); 
        console.log("%d XOR %d => %d", a, b, prediction);
      });
    });
  });
});

/* OUTPUT 
0 XOR 0 => 0
0 XOR 1 => 1
1 XOR 0 => 1
1 XOR 1 => 0
*/

