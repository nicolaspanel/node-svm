/**
  Use C-SCV and RBF kernel for spam classification
  Dataset : 
   * Webb Spam Corpus unigram (http://www.cc.gatech.edu/projects/doi/WebbSpamCorpus.html)
   * # of classes: 2
   * # of data: 20,000 (choosen randomly from the official dataset)
  
  Note : 
   * nodesvm#findBestParameters function help to find gamma and C parameters
  that provide the highest f-score on the reduced dataset
   * Webb Spam dataset is already normalized so we don't normalize it again
**/
'use strict';

var nodesvm = require('../lib/nodesvm'),
    _ = require('underscore'),
    hd = require("humanize-duration"),
    fileName = './examples/datasets/webspam_unigram_subset20000.ds',
    nFold= 4,
    start = new Date();

nodesvm.readDatasetAsync(fileName, function(webbSpam){
  console.log('dataset loaded. Start dataset reduction (using PCA)..');

  var pca = nodesvm.reduceDatasetDimension(webbSpam, 0.99);
  console.log('Dataset dimensions reduced from %d to %d features.', pca.oldDimension, pca.newDimension);
  
  console.log('Evaluation started (may take a while)...');
  var options = {
    svmType: nodesvm.SvmTypes.C_SVC,
    kernelType: nodesvm.KernelTypes.RBF,
    fold: nFold,
    cValues: [0.125, 0.5, 2, 8],
    gValues: [8, 2, 0.5, 0.125]
  };
  nodesvm.findBestParameters(pca.dataset, options, function (report) {

    console.log('Evaluation result : \n', JSON.stringify(report, null, '\t'));
    console.log('Total time : %s', hd(new Date() - start));
    
  }, function(progressRate, remainingTime){
    // called during evaluation to report progress
    // remainingTime in ms
    console.log('%d% - %s remaining...', progressRate * 100, hd(remainingTime));
  });
}); 

/* OUTPUT
dataset loaded. Start dataset reduction (using PCA)..
Dataset dimensions reduced from 254 to 28 features.
Evaluation started (may take a while)...
6% - 1 hour, 1 minute, 45 seconds, 285 milliseconds remaining...
13% - 43 minutes, 15 seconds, 964 milliseconds remaining...
19% - 33 minutes, 41 seconds, 487 milliseconds remaining...
25% - 26 minutes, 50 seconds, 436 milliseconds remaining...
31% - 26 minutes, 55 seconds, 169 milliseconds remaining...
38% - 24 minutes, 43 seconds, 88 milliseconds remaining...
44% - 21 minutes, 56 seconds, 538 milliseconds remaining...
50% - 18 minutes, 53 seconds, 805 milliseconds remaining...
56% - 18 minutes, 19 seconds, 890 milliseconds remaining...
63% - 16 minutes, 17 seconds, 644 milliseconds remaining...
69% - 13 minutes, 42 seconds, 20 milliseconds remaining...
75% - 10 minutes, 54 seconds, 723 milliseconds remaining...
81% - 8 minutes, 28 seconds, 438 milliseconds remaining...
88% - 5 minutes, 21 seconds, 532 milliseconds remaining...
94% - 2 minutes, 32 seconds, 495 milliseconds remaining...
100% - 0 remaining...
Evaluation result : 
 {
  "accuracy": 0.982,
  "fscore": 0.9771953239274362,
  "C": 8,
  "gamma": 8,
  "nbIterations": 16
}
Total time : 38 minutes, 57 seconds, 214 milliseconds

*/