'use strict';

var _ = require('underscore'),
    fs = require('fs'),
    numeric = require('numeric'),
    classificator = require('./evaluators');

var readProblemAsync = function(path, callback){
  fs.readFile(path, 'utf8', function (err, data) {
    if (err) {
      throw err;
    }
    var result = [];
    var lines = data.split("\n");
    lines = _.reject(lines, function(str){ return str.trim() === ''; }); // remove empty lines
    var nbFeatures = 0;
    lines.forEach(function(line){
      var elts = line.split(" ");
      for (var i = 1; i < elts.length; i++){
        var node = elts[i].split(":");
        var index = parseInt(node[0], 10);
        if (index > nbFeatures){
          nbFeatures = index;
        }
      }
    });
    
    lines.forEach(function(line){
      var elts = line.split(" ");
      var node = [];
      node[0] = numeric.rep([nbFeatures],0);
      for (var i = 1; i < elts.length ; i++){
        var indexValue = elts[i].split(":");
        var index = parseInt(indexValue[0], 10);
        var value = parseFloat(indexValue[1]);
        node[0][index - 1] = value;
      }
      node[1] = parseFloat(elts[0]);
      result.push(node);
    });
    callback(result, nbFeatures);
  });
};

var meanNormalizeDataSet = function(args){
  var problem = args.problem ? args.problem: [];
  var X = _.map(problem, function(ex){return ex[0];});
  var n = numeric.dim(X)[0];
  if (n===0){
    throw "invalid args. problem is empty or undefined";
  }
  var m = numeric.dim(X)[1];
  
  var mu = null, sigma = null;
  if (args.mu){
    mu = args.mu;
  }else{
    mu = numeric.rep([m],0);
  }
  
  if (args.sigma){
    sigma = args.sigma;
  }else{
    sigma = numeric.rep([m],0);
  }
  // compute mu and/or sigma
  var i = 0;
  for ( i = 0 ; i< m ; i++){
    var temp = numeric.rep([m],0);
    temp[i] = 1;
    var Xi = numeric.dot(X,temp);
    
    if (!args.mu){ 
      // compute mu
      for (var j1 = 0; j1<n; j1++){ 
        mu[i] += Xi[j1];
      }
      mu[i] /= n;
    }

    if (!args.sigma){
      // compute sigma
      for (var j2 = 0; j2<n; j2++){ 
        sigma[i] += Math.pow(Xi[j2] - mu[i], 2);
      }
      sigma[i] = Math.pow(sigma[i] / n, 0.5);
    }
  }

  var sigmaInv = _.map(sigma, function(value){ return value === 0 ? 1 : 1 / value;});
  // build the new problem
  var newProblem = [];
  for (i = 0 ; i< n ; i++){
    var xNorm = numeric.mul(numeric.add(problem[i][0], numeric.neg(mu)), sigmaInv);
    newProblem.push([xNorm, problem[i][1]]);
  }
  return {
    problem: newProblem, 
    mu: mu, 
    sigma: sigma
  };
};

var meanNormalizeInput = function(input, mu, sigma){
  var sigmaInv = _.map(sigma, function(value){ return value === 0 ? 1 : 1 / value;});
  return numeric.mul(numeric.add(input, numeric.neg(mu)), sigmaInv);
};

var readAndNormalizeDatasetAsync = function(path, callback){
  readProblemAsync(path, function(problem){
    var result = meanNormalizeDataSet({problem: problem});
    callback(result);
  });
};

var performNFoldCrossValidation = function(svm, dataset, nfold, callback){
  svm.performNFoldCrossValidation(dataset, nfold, callback);
};

var evaluateSvm = function(svm, testset, callback){
  svm.evaluate(testset, callback);
};

exports.readProblemAsync = readProblemAsync;
exports.meanNormalizeDataSet = meanNormalizeDataSet;
exports.meanNormalizeInput = meanNormalizeInput;
exports.readAndNormalizeDatasetAsync = readAndNormalizeDatasetAsync;
exports.performNFoldCrossValidation = performNFoldCrossValidation;
exports.evaluateSvm = evaluateSvm;
