'use strict';

var _ = require('underscore'),
    fs = require('fs'),
    numeric = require('numeric'), 
    async = require('async'),
    humanizeDuration = require("humanize-duration"),
    classificator = require('./evaluators'),
    libsvm = require('./svm');

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

var meanNormalizeInput = function(input, mu, sigma){
  var sigmaInv = _.map(sigma, function(value){ return value === 0 ? 1 : 1 / value;});
  return numeric.mul(numeric.add(input, numeric.neg(mu)), sigmaInv);
};

var meanNormalizeDataSet = function(args){
  var problem = args.dataset ? args.dataset: [];
  var X = _.map(problem, function(ex){return ex[0];});
  var n = numeric.dim(X)[0];
  if (n===0){
    throw "invalid args. dataset is empty or undefined";
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

  // build the new problem
  var newProblem = [];
  for (i = 0 ; i< n ; i++){
    var xNorm = meanNormalizeInput(problem[i][0], mu, sigma);
    newProblem.push([xNorm, problem[i][1]]);
  }
  return {
    dataset: newProblem, 
    mu: mu, 
    sigma: sigma
  };
};

var readAndNormalizeDatasetAsync = function(path, callback){
  readProblemAsync(path, function(problem){
    var result = meanNormalizeDataSet({dataset: problem});
    callback(result);
  });
};

var performNFoldCrossValidation = function(svm, dataset, nfold, callback){
  svm.performNFoldCrossValidation(dataset, nfold, callback);
};

var evaluateSvm = function(svm, testset, callback){
  svm.evaluate(testset, callback);
};

var findBestParameters = function(dataset, args, callback){
  // default options
  var svmType = libsvm.SvmTypes.C_SVC,
      kernelType = libsvm.KernelTypes.RBF,
      fold = 4,
      cValues = _.map(_.range(-5, 15, 2), function(value){return Math.pow(2, value);}),
      gValues = _.map(_.range(3, -15, -2), function(value){return Math.pow(2, value);}),
      degreeValues = [2, 3, 4, 5],
      log = true;
  
  if (typeof dataset === 'undefined' ){
    throw "noode-svm::findBestParameters : dataset is required";
  }
  if (dataset.length === 0 ){
    throw "noode-svm::findBestParameters : Supplied dataset is empty";
  }
  if (typeof args.kernelType !== 'undefined'){
    kernelType = args.kernelType;
  }
  if (typeof args.svmType !== 'undefined'){
    svmType = args.svmType;
  }
  if (typeof args.fold !== 'undefined'){
    fold = args.fold;
  }
  if (typeof args.cValues !== 'undefined'){
    cValues = args.cValues;
  }
  if (typeof args.gValues !== 'undefined'){
    gValues = args.gValues;
  }
  if (typeof args.log !== 'undefined' ){
    log = args.log;
  }
  var nbC = cValues.length,
      nbG = gValues.length,
      i = 0, j = 0, k = 0, total = 0,
      jobs = [];
  if (nbC === 0 || nbG === 0){
    throw "Provide at least value for C  and/or one value for G";
  }
  while (i < nbC || j < nbG){
    if (i / nbC < j / nbG){
      for (k = 0; k < j; k++){
        jobs.push({c: cValues[i], g: gValues[k]});
      }
      i++;
    }
    else{
      for (k = 0; k < i; k++){
        jobs.push({c: cValues[k], g: gValues[j]});
      }
      j++;
    }
  }
  var reports = [];
  i = 0;
  total = nbC * nbG;
  var start = new Date();
  async.each(jobs, function(params, done){
    var kernel = null;
    if (kernelType === libsvm.KernelTypes.LINEAR){
      kernel = new libsvm.LinearKernel();
    }
    else if (kernelType === libsvm.KernelTypes.POLY){
      kernel = new libsvm.PolynomialKernel(params.degree, params.g, params.r);
    }
    else if (kernelType === libsvm.KernelTypes.RBF){
      kernel = new libsvm.RadialBasisFunctionKernel(params.g);
    } 
    else if (kernelType === libsvm.KernelTypes.SIGMOID){
      kernel = new libsvm.SigmoidKernel(params.g, params.r);
    } 
    var svm = new libsvm.SVM({
      type: svmType,
      kernel: kernel,
      C: params.c
    });
    svm.performNFoldCrossValidation(dataset, fold, function(report){
      reports.push({
        accuracy: report.accuracy,
        fscore: report.fscore,
        gamma: params.g,
        C: params.c
      });
      i++;
      if (i%5 === 0 && log){
        var duration = (new Date()).getTime() - start.getTime();
        console.log('%d% achived. %s remaining...', Math.round(i/total * 100), humanizeDuration(duration * (total -i) / i));
      }
      done();
    });

  }, function(){
    var best = _.max(reports, function(r){ return r.fscore; });
    best['nbIterations'] = i;
    if (log){
      console.log('best params : \n', best);
    }
    callback(best);
  });
};

exports.readProblemAsync = readProblemAsync;
exports.meanNormalizeDataSet = meanNormalizeDataSet;
exports.meanNormalizeInput = meanNormalizeInput;
exports.readAndNormalizeDatasetAsync = readAndNormalizeDatasetAsync;
exports.performNFoldCrossValidation = performNFoldCrossValidation;
exports.evaluateSvm = evaluateSvm;
exports.findBestParameters = findBestParameters;
