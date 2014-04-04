'use strict';

var util = require('util'),
    events  = require('events'),
    _ = require('underscore'),
    numeric = require('numeric'),
    async = require('async');

/**
Classification Evaluator aims to check your classificator's accuracy/precision/recall/fscore.

NOTICE : ClassificationEvaluator assumes your inputs  
are properly normalized between 0 and 1 
*/
var ClassificationEvaluator = function  (classifier) {
  events.EventEmitter.call(this);
  this.classifier = classifier;
};
util.inherits(ClassificationEvaluator, events.EventEmitter);
ClassificationEvaluator.prototype._computeFScore = function(precision, recall) {
  if (recall === 0 && precision === 0){
    return 0;
  } 
  return 2 * recall * precision / (recall + precision);
};

ClassificationEvaluator.prototype._trainAndEvaluateSet = function(set) {
  var classifier = _.clone(this.classifier);
  if (set.trainning.length > 0){
    classifier.train(set.trainning);
  }
  var labels = _.uniq(_.map(set.test, function(ex){ return ex[1];}));
  
  var nbLabels = labels.length;
  var nbExamples = set.test.length;
  var results = numeric.rep([nbLabels,nbLabels],0);
  var onces = numeric.rep([nbLabels,1],1);

  var getIndex = function(label){
    var index = _.indexOf(labels, label);
    if (index === -1){ //label existing in trainning set but not in test set
      labels.push(label);
      index = nbLabels;
      _.range(nbLabels).forEach(function(i){
        results[i][index] = 0;
      });
      nbLabels++;
      results[index] = numeric.rep([1,nbLabels],0);
    }
    return index;
  };
  
  set.test.forEach(function(example){
    var prediction = classifier.predict(example[0]);
    results[getIndex(prediction)][getIndex(example[1])] += 1;
  });

  var sumPredictions = numeric.dot(results, onces);
  var sumExpected = numeric.dot(numeric.transpose(results) , onces);

  var perClassReports = [];
  var nbGoodPrediction = 0;
  
  var self = this;
  labels.forEach(function (label) {
    var labelIndex = getIndex(label);
    var TP = results[labelIndex][labelIndex];
    
    var precision = 0;
    var recall = 0;
    if (TP !== 0){
      precision =  TP / sumPredictions[labelIndex];
      recall = TP / sumExpected[labelIndex];
    }
      
    nbGoodPrediction+=TP;
    var classResult = {
      class: label,
      precision: precision,
      recall: recall,
      fscore: self._computeFScore(precision, recall)
    };
    perClassReports.push(classResult);
  });
  var classReports = {};
  perClassReports.forEach(function (r) {
    classReports[r.class] = {
      precision: r.precision,
      recall: r.recall,
      fscore: r.fscore
    };
  });
  return {
    accuracy: nbGoodPrediction / nbExamples,
    lowestFscore: _.min(_.pluck(perClassReports, 'fscore')),
    lowestPrecision: _.min(_.pluck(perClassReports, 'precision')),
    lowestRecall: _.min(_.pluck(perClassReports, 'recall')),
    classReports: classReports
  };
};
  
  
/**
NOTICE : this function assumes your classifier is already trained 
*/
ClassificationEvaluator.prototype.evaluate = function(testSet, callback){
  if (typeof callback === 'function'){
    this.once('done', callback);
  }
  this.performKFoldCrossValidation(1, testSet);
};

ClassificationEvaluator.prototype.performKFoldCrossValidation = function(kfold, dataSet, callback) {
  if (typeof callback === 'function'){
    this.once('done', callback);
  }
  var data = _.shuffle(dataSet);

  var nbExamplePerSubset = Math.floor(data.length / kfold);
  var dataSubsets = [];
  var k = 0;
  var i = 0, j=0;
  for (i = 0; i < kfold; i++){
    var subset = data.slice(k, k + nbExamplePerSubset);
    k += nbExamplePerSubset;
    dataSubsets.push(subset);
  }
  
  var _sets = [];
  for (i = 0; i < kfold; i++){
    var iTrainningSet = [];
    
    for (j = 0; j < kfold; j++){
      if(j !== i){
        iTrainningSet = iTrainningSet.concat(dataSubsets[j]);
      }
    }
    
    _sets.push({ 
      trainning: iTrainningSet,
      test: dataSubsets[i]
    });  
  }

  var setReports = [];
  var self = this;
  async.each(_sets, function(set, done){
    var r = self._trainAndEvaluateSet(set);
    setReports.push(r);
    done();
  }, function(err){
    var sumAccuracies = 0, 
        sumFScores = 0,
        sumPrecisions = 0,
        sumRecall = 0;
    setReports.forEach(function(r){
      sumAccuracies+=r.accuracy;
      sumFScores += r.lowestFscore;
      sumPrecisions += r.lowestPrecision;
      sumRecall += r.lowestRecall;
    });
    self.emit('done', {
      kfold: kfold,
      accuracy: sumAccuracies / kfold,
      fscore: sumFScores / kfold,
      precision: sumPrecisions / kfold,
      recall: sumRecall / kfold,
      subsetsReports: setReports
    });
  });  
};    
exports.ClassificationEvaluator = ClassificationEvaluator;