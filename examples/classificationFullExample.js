/**
  Use C-SCV and RBF kernel for spam classification
  Dataset : 
   * Webb Spam Corpus unigram (http://www.cc.gatech.edu/projects/doi/WebbSpamCorpus.html)
   * # of classes: 2
   * # of data: 50,000 (choosen randomly from the official dataset)
  
  Note : nodesvm#findBestParameters function help to find gamma, C and epsilon parameters
  that provide the lowest f-score
**/
'use strict';

var nodesvm = require('../lib/nodesvm'),
    _ = require('underscore'),
    humanizeDuration = require("humanize-duration"),
    fileName = './examples/datasets/webspam_unigram_subset10000.ds',
    nFold= 4,
    start = new Date();

nodesvm.readDatasetAsync(fileName, function(webbSpam){
  console.log('dateset loaded. Start dataset reduction..');
  var cSvm = new nodesvm.SVM({
    type: nodesvm.SvmTypes.C_SVC,
    kernel: new nodesvm.RadialBasisFunctionKernel(2.0),
    C: 2.0,
    cacheSize: 500
  });
  var pca = nodesvm.reduceDatasetDimension(webbSpam, 0.99);
  console.log('Dataset dimensions reduced from %d to %d features. Start cross validation...', pca.oldDimension, pca.newDimension);

  cSvm.performNFoldCrossValidation(pca.dataset, nFold, function(report){
    console.log('%d-fold CV achived in %s.\nResult : %s\n', nFold, humanizeDuration(new Date() - start), JSON.stringify(report, null, '\t')); 
  });

}); 

/* OUTPUT
dateset loaded. Start dataset reduction..
Dataset dimensions reduced from 128 to 27 features. Start cross validation...
CV achived in 29 seconds, 798 milliseconds.
Result : 
{
  "nfold": 4,
  "accuracy": 0.9352,
  "fscore": 0.9158621765788221,
  "precision": 0.9252217224140777,
  "recall": 0.9067028073041905,
  "subsetsReports": [...]
}

*/