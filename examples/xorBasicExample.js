/**
  Simple example using C-SVC classificator to predict the xor function
  Dataset : xor problem loaded from file (libsvm format)
  
  Note : because dataset is normalized we also normalize inputs
**/

var libsvm = require('../lib/nodesvm');
var fileName = './examples/datasets/xor.ds';
var svm = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(0.5),
  C: 1
});

libsvm.readAndNormalizeDatasetAsync(fileName, function(xor){ 
  var mu = xor.mu,
      sigma = xor.sigma;
  
  svm.trainAsync(xor.problem, function() {
    [0,1].forEach(function(a){
      [0,1].forEach(function(b){
        var normalizedInput = libsvm.meanNormalizeInput([a, b], mu, sigma);
        var prediction = svm.predict(normalizedInput); 
        console.log("%d XOR %d => %d", a, b, prediction);
      });
    });
  });
});

/* result : 
0 XOR 0 => 0
0 XOR 1 => 1
1 XOR 0 => 1
1 XOR 1 => 0
*/

