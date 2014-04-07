/** 
  Perform C-SVC classification as describe on the libsvm guide 

  data set : svmguide1.m.ds (concatenation of svmguide1.ds and svmguide1.t.ds)

  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.
  NOTE : Use normalization to improve accuracy. Expecterd accuracy with default params = ~97%
*/
'use strict';

var nodesvm = require('../lib/nodesvm');

var cSvm= new nodesvm.SVM({
  type: nodesvm.SvmTypes.C_SVC,
  kernel: new nodesvm.RadialBasisFunctionKernel(2.0),
  C: 2.0
});
var nFold= 4,
    fileName = './examples/datasets/svmguide1.m.ds';

// Load problems
nodesvm.readAndNormalizeDatasetAsync(fileName, function(svmguide){ 
  // problem
  console.log('Data set normalized with following parameters :');
  console.log('  * mu = ', JSON.stringify(svmguide.mu));
  console.log('  * sigma = ', JSON.stringify(svmguide.sigma));   
  
  console.log('Cross validation... (may take few seconds)');
  cSvm.performNFoldCrossValidation(svmguide.dataset, nFold, function(report){
    console.log('Evaluation restult : \n', JSON.stringify(report, null, '\t')); 
  });

});

/* OUTPUT 
Data set normalized with following parameters :
  * mu = [29.914873006474114,102.70785030775836,0.07375309179150807,111.83370618944843]
  * sigma = [31.58188702774939,94.40219689696849,0.25212490174150815,40.213133558797814]
Cross validation... (may take few seconds)
Evaluation restult : 
 {
  "nfold": 4,
  "accuracy": 0.9770033860045146,
  "fscore": 0.9735555284898099,
  "precision": 0.9741834372873857,
  "recall": 0.9712168132371602,
  "subsetsReports": [object]
}

*/
