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
  C: 2
});

svm.train(xorProblem);
console.log("xor trainned");
xorProblem.forEach(function(ex){
  console.log("%d XOR %d => %d", ex[0][0], ex[0][1], svm.predict(ex[0]));
});