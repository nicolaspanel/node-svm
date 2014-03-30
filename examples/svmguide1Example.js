/** 
  perform C_SVC classification as describe on the guide 
  (see http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9).
*/

var async = require('async'), 
    libsvm = require('../lib/nodesvm');

var c_svc = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(2),
  C: 2.0
});
var trainingset = null, testset = null;

// load datasets 
async.parallel([
  function(callback){
    libsvm.readProblem('./examples/datasets/svmguide1.ds', function(problem){
      trainingset = problem;
      callback(null); // no error
    });
  },
  function(callback){
    libsvm.readProblem('./examples/datasets/svmguide1.t.ds', function(problem){
      testset = problem;
      callback(null); // no error
    });
  }
],
function(err, results){
  
  console.log("problems loaded. Start training...");
  
  c_svc.train(trainingset, function(){
    c_svc.getAccuracy(testset, function(accuracy){
      console.log("svm trained. Accuracy = %d%", accuracy);
    });

  });
});
