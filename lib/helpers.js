'use strict';

var _ = require('underscore'),
    fs = require('fs'),
    numeric = require('numeric'), 
    async = require('async'),
    moment = require('moment'),
    classificator = require('./evaluators'),
    nodesvm = require('./svm');

var readDatasetAsync = function(path, callback){
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
  var problem = [];
  if (args instanceof Array){
    problem = args;
  }else if (args.dataset){
    problem = args.dataset;
  }
  var X = _.map(problem, function(ex){return ex[0];});
  var n = problem.length;
  if (n===0){
    throw "invalid args. dataset is empty or undefined";
  }
  var m = numeric.dim(X)[1];
  if (m===0){
    throw "invalid args. dataset as no features";
  }
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
  readDatasetAsync(path, function(problem){
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
    svmType: args.svmType !== undefined ? args.svmType : nodesvm.SvmTypes.C_SVC,
    kernelType: args.kernelType !== undefined ? args.kernelType : nodesvm.KernelTypes.RBF,
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
  if (params.svmType === nodesvm.SvmTypes.C_SVC ||
      params.svmType === nodesvm.SvmTypes.EPSILON_SVR ||
      params.svmType === nodesvm.SvmTypes.NU_SVR){
    if (params.cValues.length === 0){
      throw 'Provide at least one value for C parameter (see \'cValues\' option).';
    }
  }
  else {
    params.cValues = [];
  }

  // parameter G used for POLY, RBF, and SIGMOID kernels
  if (params.kernelType === nodesvm.KernelTypes.POLY ||
      params.kernelType === nodesvm.KernelTypes.RBF ||
      params.kernelType === nodesvm.KernelTypes.SIGMOID){
    if (params.gValues.length === 0){
      throw 'Provide at least one value for gamma parameter (see \'gValues\' option).';
    }
  }
  else {
    params.gValues = [];
  }

  // parameter epsilon used for epsilon-SVR only
  if (params.svmType === nodesvm.SvmTypes.EPSILON_SVR){
    if (params.epsilonValues.length === 0){
      throw 'Provide at least one value for epsilon parameter (see \'epsilonValues\' option).';
    }
  }
  else {
    params.epsilonValues = [];
  }

  // parameter nu used for nu-SVC, one-class SVM, and nu-SVR
  if (params.svmType === nodesvm.SvmTypes.NU_SVC ||
      params.svmType === nodesvm.SvmTypes.ONE_CLASS ||
      params.svmType === nodesvm.SvmTypes.NU_SVR ){
    if (params.nuValues.length === 0){
      throw 'Provide at least one value for nu parameter (see \'nuValues\' option).';
    }
  }
  else {
    params.nuValues = [];
  }

  // parameter degree used only for POLY kernel
  if (params.kernelType === nodesvm.KernelTypes.POLY ){
    if (params.dValues.length === 0){
      throw 'Provide at least one value for degree parameter (see \'dValues\' option).';
    }
  }
  else {
    params.dValues = [];
  }

  // parameter r used for POLY kernel
  if (params.kernelType === nodesvm.KernelTypes.POLY  ||
      params.kernelType === nodesvm.KernelTypes.SIGMOID ){
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
  var start = moment();
  async.each(combinaisons, function(comb, done){

    var kernel = null, 
        c = comb[0], 
        g = comb[1], 
        epsilon = comb[2], 
        nu = comb[3], 
        degree = comb[4], 
        r = comb[5];
    
    if (params.kernelType === nodesvm.KernelTypes.LINEAR){
      kernel = new nodesvm.LinearKernel();
    }
    else if (params.kernelType === nodesvm.KernelTypes.POLY){
      kernel = new nodesvm.PolynomialKernel(degree, g, r);
    }
    else if (params.kernelType === nodesvm.KernelTypes.RBF){
      kernel = new nodesvm.RadialBasisFunctionKernel(g);
    } 
    else if (params.kernelType === nodesvm.KernelTypes.SIGMOID){
      kernel = new nodesvm.SigmoidKernel(g, r);
    } 
    
    var svm = new nodesvm.SVM({
      type: params.svmType,
      kernel: kernel,
      C: c,
      epsilon: epsilon,
      nu: nu
    });
    
    svm.performNFoldCrossValidation(dataset, params.fold, function(result){
      var report = {};
      if (params.svmType === nodesvm.SvmTypes.C_SVC ||
          params.svmType === nodesvm.SvmTypes.NU_SVC) {
        report['accuracy'] = result.accuracy;
        report['fscore'] = result.fscore;
      }
      else {
        report['mse'] = result.mse;
      }
      
      if (params.svmType === nodesvm.SvmTypes.C_SVC ||
          params.svmType === nodesvm.SvmTypes.EPSILON_SVR ||
          params.svmType === nodesvm.SvmTypes.NU_SVR){
        report['C'] = c;
      }

      if (params.kernelType === nodesvm.KernelTypes.POLY ||
          params.kernelType === nodesvm.KernelTypes.RBF ||
          params.kernelType === nodesvm.KernelTypes.SIGMOID){
        report['gamma'] = g;
      }

      if (params.svmType === nodesvm.SvmTypes.EPSILON_SVR){
        report['epsilon'] = epsilon;
      }

      if (params.svmType === nodesvm.SvmTypes.NU_SVC ||
          params.svmType === nodesvm.SvmTypes.ONE_CLASS ||
          params.svmType === nodesvm.SvmTypes.NU_SVR ){
        report['nu'] = nu;
      }

      if (params.kernelType === nodesvm.KernelTypes.POLY ){
        report['degree'] = degree;
      }

      if (params.kernelType === nodesvm.KernelTypes.POLY  ||
          params.kernelType === nodesvm.KernelTypes.SIGMOID ){
        report['r'] = r;
      }

      reports.push(report);
      i++;
      var progress = Math.round(i * 100 / total ) / 100;
      if (progress !== lastReportedProgress){
        var elapsed = moment().diff(start) ;
        sendProgression(progress, elapsed * (total -i) / i);
        lastReportedProgress = progress;
      }
      done();
    });

  }, function(){
    // once all combinaisons evaluated
    var best = null;
    if (params.svmType === nodesvm.SvmTypes.C_SVC ||
        params.svmType === nodesvm.SvmTypes.NU_SVC) {
      best = _.max(reports, function(r){ return r.fscore; });
    }
    else {
      best = _.min(reports, function(r){ return r.mse; });
    } 

    best['nbIterations'] = i;
    callback(best);
  });
};

var reduceInputDimension= function(input, u){
  return numeric.dot(input, u);
};

var reduceDatasetDimension = function(dataset, minimumVarianceRetained){
  if (dataset.length === 0){
    throw 'nodesvm#reduceDatasetDimension :: Invalid datatset';
  }
  if (typeof minimumVarianceRetained === 'undefined'){
    minimumVarianceRetained = 0.99;
  }
  var inputs =  _.map(dataset, function(ex){ return ex[0]; });
  var covMatrix  = numeric.dot(numeric.transpose(inputs),inputs);
  covMatrix = numeric.mul(covMatrix, numeric.rep(numeric.dim(covMatrix), 1 / inputs.length));
  var usv = numeric.svd(covMatrix);
  
  var getFirstColumns = function(matrix, nbColumns){
    return _.map(matrix, function(line) { 
      var newLine = [];
      for (var i = 0; i < nbColumns; i++){
        newLine.push(line[i]); 
      }
      return newLine;
    });
  };
  var n = inputs[0].length,
      k = inputs[0].length,
      j = 0, retain = 1;
  
  while (true){
    // decrease k while retain variance is acceptable
    var num = 0;
    var den = 0;
    for (j = 0; j<n; j++){
      if (j < k){
        num += usv.S[j];
      }
      den += usv.S[j];
    }
    var newRetain = num / den;
    if (newRetain < minimumVarianceRetained || k === 0){
      k++;
      break;
    }
    retain = newRetain;
    k--;
  }
  var reducedU = getFirstColumns(usv.U, k);
  // compute new dataset
  var newDataset = _.map(dataset, function(ex){
    var input = reduceInputDimension(ex[0], reducedU);
    return [input, ex[1]];
  });

  return {
    U: reducedU,
    oldDimension: n,
    newDimension: k,
    dataset: newDataset,
    retainedVariance: retain
  };
};

var createLibsvmModelFileContent = function(args){
  var content = [];
  var svmTypes = _.map(nodesvm.SvmTypes, function(v, k){ return k;});
  content.push('svm_type ' + svmTypes[args.svmType].toLowerCase());
  switch (args.kernelType){
  case nodesvm.KernelTypes.RBF : 
    content.push('kernel_type rbf');
    break;
  case nodesvm.KernelTypes.SIGMOID : 
    content.push('kernel_type sigmoid');
    break;
  case nodesvm.KernelTypes.POLY : 
    content.push('kernel_type polynomial');
    break;
  case nodesvm.KernelTypes.LINEAR : 
    content.push('kernel_type linear');
    break;
  default:
    throw '#nodesvm#createLibsvmModelFileContent :: \'' + args.kernelType +'\' is not a valid kernel type';
  }
  if (args.gamma){
    content.push('gamma ' + args.gamma);
  }
  switch (args.SvmTypes){
  case nodesvm.KernelTypes.NU_SVC : 
  case nodesvm.KernelTypes.C_SVC : 
    content.push('nr_class ' + args.classes.length);
    break;
  default:
    content.push('nr_class 2');
  }
  content.push('total_sv ' + args.supportVectors.length);
  content.push('rho ' + args.rho);
  content.push('label ' + args.classes.join(' '));
  if (args.probA){
    content.push('probA ' + args.probA);
  }
  if (args.probB){
    content.push('probB ' + args.probB);
  }

  content.push('nr_sv ' + args.nbSV.join(' '));

  content.push('SV');
  args.supportVectors.forEach(function(ex){
    var inputs = [];
    for (var i =0; i < ex[0].length; i++){
      inputs.push( i+1 + ':' + ex[0][i]);
    }
    inputs.push('');
    content.push(ex[1] + ' ' + inputs.join(' '));
  });
  content.push('');
  return content.join('\n');
};
var readLibsvmModelFileContent = function(content){
  var values = {};
  var lines = content.split('\n');
  var i = 0, j=0, k=0;
  while (i < lines.length){
    var args = lines[i].split(' ');
    if (args[0] === 'SV'){
      i++;
      break;
    }
    if (args[0] === 'svm_type'){
      switch(args[1]){
      case 'c_svc' : 
        values['svmType'] = nodesvm.SvmTypes.C_SVC;
        break;
      case 'nu_svc' : 
        values['svmType'] = nodesvm.SvmTypes.NU_SVC;
        break;
      case 'nu_svr' : 
        values['svmType'] = nodesvm.SvmTypes.NU_SVR;
        break;
      case 'epsilon_svr' : 
        values['svmType'] = nodesvm.SvmTypes.EPSILON_SVR;
        break;
      default : 
        throw 'ONE CLASS SVM not supported';
      }
    }

    if (args[0] === 'kernel_type'){
      switch(args[1]){
      case 'rbf' : 
        values['kernelType'] = nodesvm.KernelTypes.RBF;
        break;
      case 'sigmoid' : 
        values['kernelType'] = nodesvm.KernelTypes.RBF;
        break;
      case 'polynomial' : 
        values['kernelType'] = nodesvm.KernelTypes.RBF;
        break;
      case 'linear' : 
        values['kernelType']= nodesvm.KernelTypes.RBF;
        break;
      }
    }

    if (args[0] === 'gamma'){
      values['gamma'] = parseFloat(args[1], 10);
    }
    if (args[0] === 'probA'){
      values['probA'] = parseFloat(args[1], 10);
    }
    if (args[0] === 'probB'){
      values['probB'] = parseFloat(args[1], 10);
    }
    if (args[0] === 'nr_class'){
      values['nbClasses'] = parseInt(args[1], 10);
    }
    if (args[0] === 'rho'){
      values['rho'] = parseFloat(args[1], 10);
    }
    if (args[0] === 'label'){
      values['classes'] = [];
      for (j = 1; j < args.length; j++) {
        values['classes'].push(parseFloat(args[j], 10));
      }
    }
    if (args[0] === 'nr_sv'){
      values['nbSV'] = [];
      for (j = 1; j < args.length; j++) {
        values['nbSV'].push(parseFloat(args[j], 10));
      }
    }
    i++;
  }
  var supportVectors = [];
  for (j = i; j < lines.length; j++){
    if (lines[j] === ''){
      break;
    }
    var line = lines[j].split(' ');
    
    var y = parseFloat(line[0], 10);
    var maxIndex = 0,
        inputs = [],
        node = null,
        index = 0,
        value = 0;
    
    for (k = 1; k< line.length; k++){
      if (line[k] === ''){
        break;
      }
      node = line[k].split(':');
      index = parseInt(node[0], 10);
      if (index > maxIndex){
        maxIndex = index;
      }
    }
    inputs = numeric.rep([maxIndex],0);
    for (k = 1; k< line.length; k++){
      if (line[k] === ''){
        break;
      }
      node = line[k].split(':');
      index = parseInt(node[0], 10);
      value = parseFloat(node[1], 10);
      inputs[index-1] = value;
    }
    supportVectors.push([inputs, y]);
  }
  values['supportVectors'] = supportVectors;
  return values;
};


exports.readDatasetAsync = readDatasetAsync;
exports.meanNormalizeDataSet = meanNormalizeDataSet;
exports.meanNormalizeInput = meanNormalizeInput;
exports.readAndNormalizeDatasetAsync = readAndNormalizeDatasetAsync;
exports.performNFoldCrossValidation = performNFoldCrossValidation;
exports.evaluateSvm = evaluateSvm;
exports.findAllPossibleCombinaisons = findAllPossibleCombinaisons;
exports.findBestParameters = findBestParameters;
exports.reduceInputDimension = reduceInputDimension;
exports.reduceDatasetDimension = reduceDatasetDimension;
exports.createLibsvmModelFileContent = createLibsvmModelFileContent;
exports.readLibsvmModelFileContent = readLibsvmModelFileContent;
