/**
  Use C-SCV and RBF kernel for spam classification
  Dataset : 
   * Webb Spam Corpus unigram (http://www.cc.gatech.edu/projects/doi/WebbSpamCorpus.html)
   * # of classes: 2
   * # of data: 10,000 (choosen randomly from the official dataset)
  
  Note : 
   * nodesvm#findBestParameters function help to find gamma and C parameters
  that provide the highest f-score on the reduced dataset
   * Webb Spam dataset is already normalized so we don't normalize it again
**/
'use strict';

var nodesvm = require('../lib/nodesvm'),
    _ = require('underscore'),
    hd = require("humanize-duration"),
    fileName = './examples/datasets/webspam_unigram_subset10000.ds',
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
    
  }, function(progressRate, remainingTime){
    // called during evaluation to report progress
    // remainingTime in ms
    console.log('%d% - %s remaining...', progressRate * 100, hd(remainingTime));
  });
}); 

/* OUTPUT
dataset loaded. Start dataset reduction (using PCA)..
Dataset dimensions reduced from 128 to 27 features.
Evaluation started (may take a minute)...
6% - 18 minutes, 19 seconds, 335 milliseconds remaining...
13% - 12 minutes, 5 seconds, 249 milliseconds remaining...
19% - 9 minutes, 10 seconds, 901 milliseconds remaining...
25% - 7 minutes, 17 seconds, 595 milliseconds remaining...
31% - 7 minutes, 23 seconds, 913 milliseconds remaining...
38% - 6 minutes, 44 seconds, 825 milliseconds remaining...
44% - 5 minutes, 50 seconds, 580 milliseconds remaining...
50% - 4 minutes, 58 seconds, 290 milliseconds remaining...
56% - 4 minutes, 54 seconds, 782 milliseconds remaining...
63% - 4 minutes, 23 seconds, 199 milliseconds remaining...
69% - 3 minutes, 40 seconds, 184 milliseconds remaining...
75% - 2 minutes, 54 seconds, 469 milliseconds remaining...
81% - 2 minutes, 19 seconds, 673 milliseconds remaining...
88% - 1 minute, 28 seconds, 462 milliseconds remaining...
94% - 41 seconds, 956 milliseconds remaining...
Evaluation result : 
 {
  "accuracy": 0.9799,
  "fscore": 0.9739484449463187,
  "C": 8,
  "gamma": 8,
  "nbIterations": 16
}
*/