var libsvm = require('../lib/nodesvm');

var svm = new libsvm.SVM({
  file: './examples/models/xor.model'
});

console.log("xor trainned");
[
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
].forEach(function(ex){
  console.log("%d XOR %d => %d", ex[0][0], ex[0][1], svm.predict(ex[0]));
});