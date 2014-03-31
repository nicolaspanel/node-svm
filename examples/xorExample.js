var libsvm = require('../lib/nodesvm');

var xorProblem = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 }
];

var svm = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(2),
  C: 2
});

svm.train(xorProblem);
console.log("xor trainned");
xorProblem.forEach(function(ex){
  console.log("%d XOR %d => %d", ex.x[0], ex.x[1], svm.predict(ex.x));
});