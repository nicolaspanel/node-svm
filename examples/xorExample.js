/**
  Simple example using C-SVC classificator to predict the xor function 
  Note : XOR is non-linear. 
**/

var libsvm = require('../lib/nodesvm');

var xorProblem = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
];
var xorNormProblem = [
  [[-1, -1], 0],
  [[-1,  1], 1],
  [[ 1, -1], 1],
  [[ 1,  1], 0]
];

var svm = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(0.5),
  C: 1
});

svm.train(xorNormProblem);
console.log("xor trainned");
xorNormProblem.forEach(function(ex){
  console.log("%d XOR %d => %d", ex[0][0], ex[0][1], svm.predict(ex[0]));
});