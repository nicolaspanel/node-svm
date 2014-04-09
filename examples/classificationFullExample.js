/**
  Use C-SCV and RBF kernel for spam classification
  Dataset : 
   * Webb Spam Corpus unigram (http://www.cc.gatech.edu/projects/doi/WebbSpamCorpus.html)
   * # of classes: 2
   * # of data: 20,000 (choosen randomly from the official dataset)
  
  Note : 
   * Webb Spam dataset is already normalized so we don't normalize it again
   * node-svm use PCA reduction by default to speed-up training . This is why dataset is reduced from 126 to 28 features. To disable PCA set 'reduce'
   parameter to false
**/
'use strict';

var nodesvm = require('../lib/nodesvm'),
    _ = require('underscore'),
    hd = require("humanize-duration"),
    fileName = './examples/datasets/webspam_unigram_subset20000.ds',
    nFold= 4,
    start = new Date();

var svm = new nodesvm.CSVC({
  kernelType: nodesvm.KernelTypes.RBF,
  gamma: [8, 2, 0.5],
  C: [0.5, 2, 8],
  nFold: 3,
  normalize: false,
  reduce: true, // default value
  retainedVariance: 0.99 // default value 
});

svm.on('training-progressed', function (progressRate, remainingTime){
  console.log('%d% - %s remaining...', progressRate * 100, hd(remainingTime));
});

svm.once('dataset-reduced', function(oldDim, newDim, retainedVar){
  console.log('Dataset dimensions reduced from %d to %d features using PCA.', oldDim, newDim);
  console.log('%d% of the variance have been retained.', retainedVar* 100);
});

svm.once('trained', function(report){
  console.log('SVM trained. report :\n%s', JSON.stringify(report, null, '\t'));
  console.log('Total trainineg time : %s', hd(new Date() - start));
});

console.log('Start training. May take a while...');
svm.trainFromFile(fileName);

/* OUTPUT
Dataset dimensions reduced from 254 to 28 features using PCA.
99.03% of the variance have been retained.
11% - 25 minutes, 13 seconds, 920 milliseconds remaining...
22% - 17 minutes, 15 seconds, 401 milliseconds remaining...
33% - 10 minutes, 34 seconds, 626 milliseconds remaining...
44% - 9 minutes, 38 seconds, 975 milliseconds remaining...
56.00000000000001% - 7 minutes, 29 seconds, 788 milliseconds remaining...
67% - 4 minutes, 54 seconds, 1 millisecond remaining...
78% - 2 minutes, 56 seconds, 960 milliseconds remaining...
89% - 1 minute, 20 seconds, 380 milliseconds remaining...
100% - 0 remaining...
SVM trained. report :
{
  "accuracy": 0.98004800480048,
  "fscore": 0.9746267648828036,
  "C": 8,
  "gamma": 8,
  "nbIterations": 9
}
Total trainineg time : 11 minutes, 48 seconds, 19 milliseconds

*/