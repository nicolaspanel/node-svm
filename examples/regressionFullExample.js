/**
  Simple example using EPSILON-SVR classificator to predict values
  Dataset : Housing, more info here http://archive.ics.uci.edu/ml/datasets/Housing
  
  Note : libsvm#findBestParameters function help to find gamma, C and epsilon parameters
  that provie the lowest Mean Square Error () 
**/
'use strict';

var libsvm = require('../lib/nodesvm'),
    _ = require('underscore'),
    humanizeDuration = require("humanize-duration");

var nFold= 3,
    fileName = './examples/datasets/housing.ds';

libsvm.readAndNormalizeDatasetAsync(fileName, function(housing){ 
  console.log('Data set normalized with following parameters :');
  console.log('  * mu = \n', JSON.stringify(housing.mu));
  console.log('  * sigma = \n', JSON.stringify(housing.sigma));   

  console.log('Looking for parameters that provide the lower Mean Square Error : ');
  var args = {
    svmType : libsvm.SvmTypes.EPSILON_SVR,
    kernelType : libsvm.KernelTypes.RBF,
    cValues: [0.03125, 0.125, 0.5, 2, 8],
    gValues: [8, 2, 0.5, 0.125, 0.03125],
    epsilonValues: [8, 2, 0.5, 0.125, 0.03125],
    fold: nFold
  }; 
  libsvm.findBestParameters(housing.dataset, args, function(report) {
    // build SVM with found parameters
    console.log('Best params : \n', JSON.stringify(report, null, '\t'));

    var svm = new libsvm.SVM({
      type: libsvm.SvmTypes.EPSILON_SVR,
      kernel: new libsvm.RadialBasisFunctionKernel(report.gamma),
      C: report.C,
      epsilon: report.epsilon
    });
    var training = _.sample(housing.dataset, housing.dataset.length);
    var tests = _.sample(housing.dataset, 20);
    // train the svm
    svm.trainAsync(training, function(){
      // predict some values
      for (var i = 0; i < 20;  i++){
        var test = tests[i];
        console.log('{expected: %d, predicted: %d}', test[1], svm.predict(test[0]));
      }
    });
  }, function(progressRate, remainingTime){
    // called during evaluation to report progress
    // remainingTime in ms
    if ((progressRate*100)%10 === 0){
      console.log('%d% achived. %s remaining...', progressRate * 100, humanizeDuration(remainingTime));
    }
  }); 
});

/* OUTPUT
Data set normalized with following parameters :
  * mu = 
 [3.6135235573122535,11.363636363636363,11.136778656126504,0.0691699604743083,0.5546950592885372,6.284634387351787,68.57490118577078,3.795042687747034,9.549407114624506,408.2371541501976,18.455533596837967,356.67403162055257,12.653063241106723]
  * sigma = 
 [8.593041351295769,23.299395694766027,6.853570583390873,0.25374293496034855,0.11576311540656153,0.7019225143345692,28.121032570236885,2.103628356344459,8.698651117790645,168.3704950393814,2.162805191482142,91.20460745217272,7.134001636650485]
Looking for parameters that provide the lower Mean Square Error : 
10% achived. 39 seconds, 32 milliseconds remaining...
20% achived. 24 seconds, 904 milliseconds remaining...
30% achived. 19 seconds, 212 milliseconds remaining...
40% achived. 15 seconds, 238 milliseconds remaining...
50% achived. 12 seconds, 209 milliseconds remaining...
60% achived. 9 seconds, 416 milliseconds remaining...
70% achived. 7 seconds, 41 milliseconds remaining...
80% achived. 4 seconds, 585 milliseconds remaining...
90% achived. 2 seconds, 415 milliseconds remaining...
100% achived. 0 remaining...
Best params : 
 {
  "mse": 11.882979662174995,
  "C": 8,
  "gamma": 0.125,
  "epsilon": 0.125,
  "nbIterations": 125
}
{expected: 17.6, predicted: 17.938559007483047}
{expected: 27.5, predicted: 23.053819272429315}
{expected: 24.4, predicted: 23.334700825652597}
{expected: 16.8, predicted: 19.461225554210092}
{expected: 35.1, predicted: 35.1001290483135}
{expected: 24.4, predicted: 24.54038631351429}
{expected: 30.7, predicted: 30.700013436030964}
{expected: 23.3, predicted: 23.299922076797852}
{expected: 16.7, predicted: 16.307359478272797}
{expected: 13.8, predicted: 13.800024447011378}
{expected: 23.8, predicted: 22.28298226432564}
{expected: 7, predicted: 10.337542018020212}
{expected: 23.9, predicted: 23.899951097477558}
{expected: 15.7, predicted: 16.5792831598902}
{expected: 10.9, predicted: 11.551552776775306}
{expected: 24.5, predicted: 27.181303003816108}
{expected: 19.9, predicted: 18.98504919924254}
{expected: 21.1, predicted: 20.4979430531197}
{expected: 14, predicted: 14.000153354646748}
{expected: 22, predicted: 22.768265406597585}
*/