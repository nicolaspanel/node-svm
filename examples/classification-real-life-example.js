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
var numeric = require('numeric');

var clf = new svm.CSVC({
    gamma: [0.01, 0.1],
    c: 8,
    kFold: 4,
    normalize: true,
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
        console.log('SVM trained. \nReport:\n%s', so(report));
    }).done(function () {
        console.log('done.');
    });

/*************************
 *        OUTPUT         *
 *************************

 start training (may take a while)...
 [training progress...]
 SVM trained.
 Report:
 {
     accuracy: 0.97485,
     fscore: 0.9681221877178527,
     recall: 0.9694123619748699,
     precision: 0.9668354430379746,
     class: {
         '1': {
             precision: 0.9800826446280991,
             recall: 0.9783846217308803,
             fscore: 0.979232897072788,
             size: 12121
         },
         '-1': {
             precision: 0.9668354430379746,
             recall: 0.9694123619748699,
             fscore: 0.9681221877178527,
             size: 7879
         }
     },
     size: 20000,
     reduce: true,
     retainedVariance: 0.950955024168114,
     retainedDimension: 62,
     initialDimension: 254
 }
 done.

 */