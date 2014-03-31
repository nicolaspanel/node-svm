/** 
  perform C_SVC classification as describe on the guide 
  (see http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9).
*/

var libsvm = require('../lib/nodesvm');

var c_svc = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(0.5),
  C: 1
});

// // load svmgui1 problem
// libsvm.readAndNormalizeProblemAsync('./examples/datasets/svmguide1.ds', function(svmguide1){    
//   // load test set and normalize it with previous mu and sigma
//   var trainigset = svmguide1.problem;
  
//   libsvm.readProblemAsync('./examples/datasets/svmguide1.t.ds', function(testset){
//     var normTestset = libsvm.meanNormalize({problem: testset, mu: svmguide1.mu, sigma: svmguide1.sigma}).problem;
//     console.log("problems loaded. Start training...");
//     c_svc.train(trainigset, function(){    
//       console.log("svm trained. Ask for accuracy...");
//       c_svc.getAccuracy(normTestset, function(accuracy){
//         console.log("Accuracy = %d%", accuracy);
//       });
//     });
//   });
// });
  
