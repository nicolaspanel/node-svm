/**
  Simple example using C-SVC classificator to predict the xor function
  Dataset : xor problem loaded from file (libsvm format)
  Parameters : 
   * C: 1
   * gamma: 0.5 
   * because XOR dataset is to small, we set nFold to 1 to avoid cross validation
  Note : 
**/
'use strict';

var nodesvm = require('../lib/nodesvm'),
    fileName = './examples/datasets/xor.ds';

var svm = new nodesvm.CSVC({
  kernelType: nodesvm.KernelTypes.RBF,
  gamma: 0.5,
  C: 1,
  nFold: 1,
  normalize: false,
  reduce: false
});

svm.once('trained', function(report) {
  console.log('SVM trained. report :\n%s', JSON.stringify(report, null, '\t'));
  console.log('Lets predict XOR values');
  
  [0,1].forEach(function(a){
    [0,1].forEach(function(b){
      var prediction = svm.predict([a, b]); 
      console.log("%d XOR %d => %d", a, b, prediction);
    });
  });
	process.exit(0);
});

svm.trainFromFile(fileName);

/* OUTPUT 
SVM trained. report :
{
  "accuracy": 1,
  "fscore": 1,
  "C": 1,
  "gamma": 0.5,
  "nbIterations": 1
}
Lets predict XOR values
0 XOR 0 => 0
0 XOR 1 => 1
1 XOR 0 => 1
1 XOR 1 => 0
*/

