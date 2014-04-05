/** 
  Perform C_SVC classification as describe on the libsvm guide 

  data set : svmguide1.m.ds, concatenation of svmguide1.ds and svmguide1.t.ds

  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.
  NOTE : Use normalization to improve accuracy. Expecterd accuracy with default params = ~97%
*/

var libsvm = require('../lib/nodesvm');

var c_svc = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(2.0),
  C: 2.0
});
var nFold= 4,
    fileName = './examples/datasets/svmguide1.m.ds';

// Load problems
libsvm.readAndNormalizeDatasetAsync(fileName, function(svmguide){ 
  // problem
  console.log("Data set normalized with following parameters :");
  console.log("  * mu = ", svmguide.mu);
  console.log("  * sigma = ", svmguide.sigma);
  
  console.log("Cross validation... (may take few seconds)");
  c_svc.performNFoldCrossValidation(svmguide.dataset, nFold, function(report){
    console.log("Accuracy = %d%%", report.accuracy * 100);
    console.log("F-Score = %d", report.fscore);
    console.log("Precision = %d", report.precision);
    console.log("Recall = %d", report.recall);
  });

});
