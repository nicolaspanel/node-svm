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
var svm = require('../lib');
var fileName = './examples/datasets/webspam_unigram_subset20000.ds';

var clf = new svm.CSVC({
  gamma: 8,
  c: 8,
  kFold: 5,
  normalize: false,
  reduce: true, // default value
  retainedVariance: 0.95
});

svm.read(fileName)
    .then(function (dataset) {
        console.log('start training (may take a while)...');
        return clf.train(dataset)
            .progress(function(progress){
                console.log('training progress: %d%', Math.round(progress*100));
            });
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
         accuracy: 0.88245,
         fscore: 0.8487616596976519,
         recall: 0.8372889960654906,
         precision: 0.8605530915731803,
         class: {
                 '1': {
                         precision: 0.8960596724501378,
                         recall: 0.9118059566042406,
                         fscore: 0.9038642404416275,
                         size: 12121
                 },
                 '-1': {
                         precision: 0.8605530915731803,
                         recall: 0.8372889960654906,
                         fscore: 0.8487616596976519,
                         size: 7879
                 }
         },
         size: 20000,
         retainedVariance: 0.950256371209233
 }
 done.

*/