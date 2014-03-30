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
libsvm.readProblem('./examples/datasets/svmguide1.ds', function(trainingset){
  
  libsvm.meanNormalize({problem: trainingset}, function(normTrainingset, mu, sigma){
    
    // load test set and normalize it with previous mu and sigma
    libsvm.readProblem('./examples/datasets/svmguide1.t.ds', function(testset){
      libsvm.meanNormalize({testset: testset, mu: mu, sigma: sigma}, function(normTestset){
        
        console.log("problems loaded. Start training...");
        c_svc.train(trainingset, function(){
          
          c_svc.getAccuracy(testset, function(accuracy){
            
            console.log("svm trained. Accuracy = %d%", accuracy);
          
          });

        });

      })
    });
  })
});
  
