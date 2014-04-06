'use strict';

var util = require('util'),
    events  = require('events'),
    _ = require('underscore'),
    numeric = require('numeric'),
    async = require('async');

var Evaluator = function(predictor) {
  events.EventEmitter.call(this);
  this.predictor = predictor;
};
util.inherits(Evaluator, events.EventEmitter);

/**
Classification Evaluator aims to check your classificator's accuracy/precision/recall/fscore.

NOTICE : ClassificationEvaluator do NOT normalize inputs
*/
var ClassificationEvaluator = function(classificator) {
  Evaluator.call(this, classificator);
};
util.inherits(ClassificationEvaluator, Evaluator);

ClassificationEvaluator.prototype._computeFScore = function(precision, recall) {
  if (recall === 0 && precision === 0){
    return 0;
  } 
  return 2 * recall * precision / (recall + precision);
};

ClassificationEvaluator.prototype._trainAndEvaluateSet = function(set, callback) {
  var predictor = _.clone(this.predictor);
  
  var labels = _.uniq(_.map(set.test, function(ex){ return ex[1];}));
  var nbLabels = labels.length;
  var nbExamples = set.test.length;
  var results = numeric.rep([nbLabels,nbLabels],0);
  var onces = numeric.rep([nbLabels,1],1);
  
  var perClassReports = [];
  var nbGoodPrediction = 0;

  var getIndex = function(l){
    var index = _.indexOf(labels, l);
    if (index === -1){ //label existing in trainning set but not in test set
      labels.push(l);
      index = nbLabels;
      _.range(nbLabels).forEach(function(i){
        results[i][index] = 0;
      });
      nbLabels++;
      results[index] = numeric.rep([1,nbLabels],0);
    }
    return index;
  };

  var self = this;
  async.each([set], function(s, done) {
    if (s.trainning.length > 0){
      predictor.trainAsync(set.trainning, function(){
        done();
      });
    }
    else {
      // no training required
      done();
    }
  }, function () {
    // once trained
    set.test.forEach(function(example){
      var prediction = predictor.predict(example[0]);
      results[getIndex(prediction)][getIndex(example[1])] += 1;
    });    
    
    var sumPredictions = numeric.dot(results, onces);
    var sumExpected = numeric.dot(numeric.transpose(results) , onces);
    
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
    callback({
      accuracy: nbGoodPrediction / nbExamples,
      lowestFscore: _.min(_.pluck(perClassReports, 'fscore')),
      lowestPrecision: _.min(_.pluck(perClassReports, 'precision')),
      lowestRecall: _.min(_.pluck(perClassReports, 'recall')),
      classReports: classReports
    });
  });
};
  
  
/**
NOTICE : this function assumes your predictor is already trained 
*/
ClassificationEvaluator.prototype.evaluate = function(testSet, callback){
  if (typeof callback === 'function'){
    this.once('done', callback);
  }
  this.performNFoldCrossValidation(1, testSet);
};

ClassificationEvaluator.prototype.performNFoldCrossValidation = function(nfold, dataSet, callback) {
  if (typeof callback === 'function'){
    this.once('done', callback);
  }
  var data = _.shuffle(dataSet);

  var nbExamplePerSubset = Math.floor(data.length / nfold);
  var dataSubsets = [];
  var k = 0;
  var i = 0, j=0;
  for (i = 0; i < nfold; i++){
    var subset = data.slice(k, k + nbExamplePerSubset);
    k += nbExamplePerSubset;
    dataSubsets.push(subset);
  }
  
  var _sets = [];
  for (i = 0; i < nfold; i++){
    var iTrainningSet = [];
    
    for (j = 0; j < nfold; j++){
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
    self._trainAndEvaluateSet(set, function (r) {
      setReports.push(r);
      done();
    });
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
      nfold: nfold,
      accuracy: sumAccuracies / nfold,
      fscore: sumFScores / nfold,
      precision: sumPrecisions / nfold,
      recall: sumRecall / nfold,
      subsetsReports: setReports
    });
  });  
};  

/**
Classification Evaluator aims to check your classificator's accuracy/precision/recall/fscore.

NOTICE : ClassificationEvaluator do NOT normalize inputs
*/
var RegressionEvaluator = function  (predictor) {
  // Call the parent's constructor
  Evaluator.call(this, predictor);
};
util.inherits(RegressionEvaluator, Evaluator);

RegressionEvaluator.prototype._trainAndEvaluateSet = function(set, callback) {
  var predictor = _.clone(this.predictor);
  
  var nbExamples = set.test.length;

  var self = this;
  async.each([set], function(s, done) {
    if (s.trainning.length > 0){
      predictor.trainAsync(set.trainning, function(){
        done();
      });
    }
    else {
      // no training required
      done();
    }
  }, function () {
    // once trained
    var mse = 0;
    for (var i = 0; i < nbExamples; i++) {
      var example = set.test[i];
      mse +=  Math.pow(example[1] - predictor.predict(example[0]), 2);
    }
    callback({
      mse: mse / nbExamples
    });
  });
};
  
  
/**
NOTICE : this function assumes your predictor is already trained 
*/
RegressionEvaluator.prototype.evaluate = function(testSet, callback){
  if (typeof callback === 'function'){
    this.once('done', callback);
  }
  this.performNFoldCrossValidation(1, testSet);
};

RegressionEvaluator.prototype.performNFoldCrossValidation = function(nfold, dataSet, callback) {
  if (typeof callback === 'function'){
    this.once('done', callback);
  }
  var data = _.shuffle(dataSet);

  var nbExamplePerSubset = Math.floor(data.length / nfold);
  var dataSubsets = [];
  var k = 0;
  var i = 0, j=0;
  for (i = 0; i < nfold; i++){
    var subset = data.slice(k, k + nbExamplePerSubset);
    k += nbExamplePerSubset;
    dataSubsets.push(subset);
  }
  
  var _sets = [];
  for (i = 0; i < nfold; i++){
    var iTrainningSet = [];
    
    for (j = 0; j < nfold; j++){
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
    self._trainAndEvaluateSet(set, function (r) {
      setReports.push(r);
      done();
    });
  }, function(err){
    var mses = _.map(setReports, function (r) { return r.mse; });
    var sumMSE = _.reduce(mses, function (memo, val) { return memo + val; });
    self.emit('done', {
      nfold: nfold,
      mse: sumMSE / nfold,
      subsetsReports: setReports
    });
  });  
};  

exports.ClassificationEvaluator = ClassificationEvaluator;
exports.RegressionEvaluator = RegressionEvaluator;