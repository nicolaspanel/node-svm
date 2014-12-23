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

var so = require('stringify-object');
var nodesvm = require('../lib');
var fileName = './examples/datasets/webspam_unigram_subset20000.ds';
var start = new Date();

var svm = new nodesvm.CSVC({
//  gamma: 8,
//  c: 8,
  kFold: 5,
  normalize: false,
  reduce: true, // default value
  retainedVariance: 0.95
});

nodesvm.read(fileName)
    .then(function (dataset) {
        console.log('start training (may take a while)...');
        return svm.train(dataset);
    })
    .spread(function (model, report) {
        console.log('SVM trained. \nReport :\n%s', so(report));
    }).done(function () {
        console.log('done.');
    });

/*************************
 *        OUTPUT         *
 *************************

 start training (may take a while)...
 SVM trained.
 Report :
 {
     accuracy: 0.94545,
     fscore: 0.9300327069839032,
     recall: 0.9202944536108644,
     precision: 0.9399792584910552,
     class: {
         '1': {
             precision: 0.9488849096532639,
             recall: 0.9618018315320518,
             fscore: 0.9552997090998484,
             size: 12121
         },
         '-1': {
             precision: 0.9399792584910552,
             recall: 0.9202944536108644,
             fscore: 0.9300327069839032,
             size: 7879
         }
     },
     size: 20000,
     retainedVariance: 0.9903772241548948
 }
 done.

*/