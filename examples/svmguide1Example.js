/** 
  perform C_SVC classification as describe on the guide 
  (see http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9).
*/

var libsvm = require('../lib/nodesvm');

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
      libsvm.meanNormalize({problem: testset, mu: mu, sigma: sigma}, function(normTestset){
        
        console.log("problems loaded. Start training...");
        c_svc.train(normTrainingset, function(){
          
          console.log("svm trained. Ask for accuracy...");
          c_svc.getAccuracy(normTestset, function(accuracy){
            
            console.log("Accuracy = %d%", accuracy);
          
          });

        });

      })
    });
  })
});
  
