'use strict';

var _ = require('underscore'),
    fs = require('fs'),
    numeric = require('numeric'), 
    async = require('async'),
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

var findAllPossibleCombinaisons = function(params){
  var nbCombinaisons = 1;
  params.forEach(function(values){
    nbCombinaisons *= values.length > 0? values.length : 1;
  });
  var result = numeric.rep([nbCombinaisons, params.length], 0);


  var i = 0, j = 0, k = 0, l = 0;
  var duration = 1;
  for (i = 0; i < params.length; i++){
    var input = params[i];
    k = 0;
    if (input.length > 0){
      while ( k < nbCombinaisons){
        for (j = 0; j < input.length; j++){
          for (l = 0; l < duration; l++) {
            result[k][i] = input[j];
            k++;
          } 
        }
      }
      duration *= input.length ;
    }
  }
  return result;
};

var findBestParameters = function(dataset, args, callback, progressReports){
  // default options
  var params = {
    svmType: args.svmType !== undefined ? args.svmType : libsvm.SvmTypes.C_SVC,
    kernelType: args.kernelType !== undefined ? args.kernelType : libsvm.KernelTypes.RBF,
    fold: args.fold !== undefined ? args.fold : 4,
    cValues: args.cValues !== undefined ? args.cValues : [],
    gValues: args.gValues !== undefined ? args.gValues : [],
    epsilonValues : args.epsilonValues !== undefined ? args.epsilonValues : [],
    nuValues : args.nuValues !== undefined ? args.nuValues : [],
    dValues : args.dValues !== undefined ? args.dValues : [],
    rValues : args.rValues !== undefined ? args.rValues : []
  };
  
  if ( dataset === undefined ){
    throw "noode-svm::findBestParameters : dataset is required";
  }
  if (dataset.length === 0 ){
    throw "noode-svm::findBestParameters : Supplied dataset is empty";
  }

  var sendProgression = function(progress, remainingTime){
    // do nothing
  };
  if (typeof progressReports === 'function'){
    sendProgression = progressReports;
  }

  
  // parameter C used for C-SVC, epsilon-SVR, and nu-SVR
  if (params.svmType === libsvm.SvmTypes.C_SVC ||
      params.svmType === libsvm.SvmTypes.EPSILON_SVR ||
      params.svmType === libsvm.SvmTypes.NU_SVR){
    if (params.cValues.length === 0){
      throw 'Provide at least one value for C parameter (see \'cValues\' option).';
    }
  }
  else {
    params.cValues = [];
  }

  // parameter G used for POLY, RBF, and SIGMOID kernels
  if (params.kernelType === libsvm.KernelTypes.POLY ||
      params.kernelType === libsvm.KernelTypes.RBF ||
      params.kernelType === libsvm.KernelTypes.SIGMOID){
    if (params.gValues.length === 0){
      throw 'Provide at least one value for gamma parameter (see \'gValues\' option).';
    }
  }
  else {
    params.gValues = [];
  }

  // parameter epsilon used for epsilon-SVR only
  if (params.svmType === libsvm.SvmTypes.EPSILON_SVR){
    if (params.epsilonValues.length === 0){
      throw 'Provide at least one value for epsilon parameter (see \'epsilonValues\' option).';
    }
  }
  else {
    params.epsilonValues = [];
  }

  // parameter nu used for nu-SVC, one-class SVM, and nu-SVR
  if (params.svmType === libsvm.SvmTypes.NU_SVC ||
      params.svmType === libsvm.SvmTypes.ONE_CLASS ||
      params.svmType === libsvm.SvmTypes.NU_SVR ){
    if (params.nuValues.length === 0){
      throw 'Provide at least one value for nu parameter (see \'nuValues\' option).';
    }
  }
  else {
    params.nuValues = [];
  }

  // parameter degree used only for POLY kernel
  if (params.kernelType === libsvm.KernelTypes.POLY ){
    if (params.dValues.length === 0){
      throw 'Provide at least one value for degree parameter (see \'dValues\' option).';
    }
  }
  else {
    params.dValues = [];
  }

  // parameter r used for POLY kernel
  if (params.kernelType === libsvm.KernelTypes.POLY  ||
      params.kernelType === libsvm.KernelTypes.SIGMOID ){
    if (params.rValues.length === 0){
      throw 'Provide at least one value for r parameter (see \'rValues\' option).';
    }
  }else {
    params.rValues = [];
  }

  var i = 0, j = 0, k = 0, total = 0, lastReportedProgress = -1;
  var combinaisons = findAllPossibleCombinaisons([
    params.cValues,
    params.gValues,
    params.epsilonValues,
    params.nuValues,
    params.dValues,
    params.rValues
  ]);
  var reports = [];
  total = combinaisons.length;
  var start = new Date();
  async.each(combinaisons, function(comb, done){

    var kernel = null, 
        c = comb[0], 
        g = comb[1], 
        epsilon = comb[2], 
        nu = comb[3], 
        degree = comb[4], 
        r = comb[5];
    
    if (params.kernelType === libsvm.KernelTypes.LINEAR){
      kernel = new libsvm.LinearKernel();
    }
    else if (params.kernelType === libsvm.KernelTypes.POLY){
      kernel = new libsvm.PolynomialKernel(degree, g, r);
    }
    else if (params.kernelType === libsvm.KernelTypes.RBF){
      kernel = new libsvm.RadialBasisFunctionKernel(g);
    } 
    else if (params.kernelType === libsvm.KernelTypes.SIGMOID){
      kernel = new libsvm.SigmoidKernel(g, r);
    } 
    
    var svm = new libsvm.SVM({
      type: params.svmType,
      kernel: kernel,
      C: c,
      epsilon: epsilon,
      nu: nu
    });
    
    svm.performNFoldCrossValidation(dataset, params.fold, function(result){
      var report = {};
      if (params.svmType === libsvm.SvmTypes.C_SVC ||
          params.svmType === libsvm.SvmTypes.NU_SVC) {
        report['accuracy'] = result.accuracy;
        report['fscore'] = result.fscore;
      }
      else {
        report['mse'] = result.mse;
      }
      
      if (params.svmType === libsvm.SvmTypes.C_SVC ||
          params.svmType === libsvm.SvmTypes.EPSILON_SVR ||
          params.svmType === libsvm.SvmTypes.NU_SVR){
        report['C'] = c;
      }

      if (params.kernelType === libsvm.KernelTypes.POLY ||
          params.kernelType === libsvm.KernelTypes.RBF ||
          params.kernelType === libsvm.KernelTypes.SIGMOID){
        report['gamma'] = g;
      }

      if (params.svmType === libsvm.SvmTypes.EPSILON_SVR){
        report['epsilon'] = epsilon;
      }

      if (params.svmType === libsvm.SvmTypes.NU_SVC ||
          params.svmType === libsvm.SvmTypes.ONE_CLASS ||
          params.svmType === libsvm.SvmTypes.NU_SVR ){
        report['nu'] = nu;
      }

      if (params.kernelType === libsvm.KernelTypes.POLY ){
        report['degree'] = degree;
      }

      if (params.kernelType === libsvm.KernelTypes.POLY  ||
          params.kernelType === libsvm.KernelTypes.SIGMOID ){
        report['r'] = r;
      }

      reports.push(report);
      i++;
      var progress = Math.round(i/total * 100) / 100;
      if (progress !== lastReportedProgress){
        var duration = (new Date()).getTime() - start.getTime();
        sendProgression(progress, duration * (total -i) / i);
        lastReportedProgress = progress;
      }
      done();
    });

  }, function(){
    var best = null;
    if (params.svmType === libsvm.SvmTypes.C_SVC ||
        params.svmType === libsvm.SvmTypes.NU_SVC) {
      best = _.max(reports, function(r){ return r.fscore; });
    }
    else {
      best = _.min(reports, function(r){ return r.mse; });
    } 

    best['nbIterations'] = i;
    callback(best);
  });
};

exports.readProblemAsync = readProblemAsync;
exports.meanNormalizeDataSet = meanNormalizeDataSet;
exports.meanNormalizeInput = meanNormalizeInput;
exports.readAndNormalizeDatasetAsync = readAndNormalizeDatasetAsync;
exports.performNFoldCrossValidation = performNFoldCrossValidation;
exports.evaluateSvm = evaluateSvm;
exports.findAllPossibleCombinaisons = findAllPossibleCombinaisons;
exports.findBestParameters = findBestParameters;
