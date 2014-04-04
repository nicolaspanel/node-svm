/** 
  Perform C_SVC classification as describe on the libsvm guide 
  See http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.
  NOTE : No scaling / normalization used. Expecterd accuracy with default params = 66.925%
*/

var libsvm = require('../lib/nodesvm');

var c_svc = new libsvm.SVM({
  type: libsvm.SvmTypes.C_SVC,
  kernel: new libsvm.RadialBasisFunctionKernel(0.25),
  C: 1
});

// Load problems
libsvm.readProblemAsync('./examples/datasets/svmguide1.ds', function(trainigSet){    
  console.log("Training set loaded. Start training...")
  c_svc.train(trainigSet);
  // problem
  libsvm.readProblemAsync('./examples/datasets/svmguide1.t.ds', function(testset){
    console.log("Test set loaded. Start accuracy evaluation...")
    var ko = 0, total = 0;
    testset.forEach(function(ex){
      if (ex[1] !== c_svc.predict(ex[0])){
        ko++;
      }
      total++;
    });
    console.log("Accuracy = %d%%% ()", (1 - ko / total)*100);
  });
});
