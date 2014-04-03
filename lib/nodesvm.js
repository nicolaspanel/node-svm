'use strict';

var util = require('util'),
    events  = require('events'),
    _ = require('underscore'),
    fs = require('fs'),
    numeric = require('numeric'),
    addon = require('../build/Release/addon');

var SvmTypes = {
  C_SVC: 0,       // multi-class classification
  NU_SVC: 1,      // multi-class classification
  ONE_CLASS: 2,   // one-class SVM 
  EPSILON_SVR : 3,// regression
  NU_SVR: 4       // regression
};


var KernelTypes = {
  LINEAR : 0,
  POLY : 1,
  RBF  : 2,
  SIGMOID : 3
};


var Kernel = function(){
  
};
Kernel.prototype._setDefaultValues = function() {
  this.kernelType = KernelTypes.RBF; // default
  this.degree = 3;                   // default value
  this.gamma = 2;                    // default value
  this.r = 0;                        // default value
};
/** 
  Linear Kernel
  K(xi , xj ) = transpose(xi) * xj
*/
var LinearKernel = function(){
  this._setDefaultValues();
  this.kernelType = KernelTypes.LINEAR;
};
util.inherits(LinearKernel, Kernel);

/** 
  Polynomial Kernel
  K(xi , xj ) = Pow( gamma * transpose(xi) * xj + r , d )
  @degree: degree of the polynome
  @gamma: width parameter
*/
var PolynomialKernel = function(degree, gamma, r){
  this._setDefaultValues();
  this.kernelType = KernelTypes.POLY;
  this.degree = degree;
  this.gamma = gamma;
  this.r = r;
};
util.inherits(PolynomialKernel, Kernel);

/** 
  RBF Kernel
  K(xi,yi) = exp( -gamma * || x - y ||² )
  @gamma: width parameter
*/
var RadialBasisFunctionKernel = function(gamma){
  this._setDefaultValues();
  this.kernelType = KernelTypes.RBF;
  this.gamma = gamma;
};
util.inherits(RadialBasisFunctionKernel, Kernel);

/** 
  Sigmoid Kernel
  K(xi , xj ) = tanh( gamma * transpose(xi) * xj + r)
  @gamma: width parameter
  @r: r parameter
*/
var SigmoidKernel = function(gamma, r){
  this._setDefaultValues();
  this.kernelType = KernelTypes.SIGMOID;
  this.gamma = gamma;
  this.r = r;
};
util.inherits(SigmoidKernel, Kernel);

var SVM = function(args){
  events.EventEmitter.call(this);
  if (!args){
    args = {};
  }
  var svmType = 0;
  if (typeof args.type !== 'undefined'){
    svmType = args.type;
  }
  var kernel = null;
  if (typeof args.kernel === 'undefined'){
    kernel = new RadialBasisFunctionKernel(2);
  }
  else{
    kernel = args.kernel;
  }
  var C = 0;
  if (typeof args.C === 'undefined'){
    C = 0.1;
  }
  else{
    C = args.C;
  }

  this._nodeSvm = new addon.NodeSvm();
  var params = {};

  params = {
    type: svmType, // default: 0 -- C_SVC
    kernel: kernel.kernelType,  
    degree: kernel.degree,
    gamma: kernel.gamma,
    r: kernel.r,
    C: C,      // cost for C_SVC, EPSILON_SVR and NU_SVR
    nu: args.nu ? args.nu : 0.5, // for NU_SVC, ONE_CLASS SVM, and NU_SVR
    p: args.p ? args.p : 0.0, // for EPSILON_SVR 
    
    eps: args.eps ? args.eps : 1e-3, // stopping criteria 
    cacheSize: args.cacheSize ? args.cacheSize: 100,                 // in MB 
    shrinking   : 1, // always use the shrinking heuristics
    probability : 1 // always true
  };
  
  var error = this._nodeSvm.setParameters(params);
  if (error){
    throw "Invalid parameter. Err: " + error;
  }
  // load params from the C++ obj.
};
util.inherits(SVM, events.EventEmitter);

SVM.prototype.train = function(problem) {  
  this._nodeSvm.train(problem);
  this.labels = this._nodeSvm.getLabels();
};

SVM.prototype.predict = function(data) {  
  return this._nodeSvm.predict(data);
};
SVM.prototype.predictAsync = function(data, callback) {  
  return this._nodeSvm.predictAsync(data, callback);
};

SVM.prototype.predictProbabilities = function(data) {
  var probs = this._nodeSvm.predictProbabilities(data);
  var result = {};
  for (var i = 0; i < probs.length ; i++){
    result[this.labels[i]] = probs[i];
  }
  return result;
};

SVM.prototype.getAccuracy = function(testData, callback) {
  this._nodeSvm.getAccuracy(testData, callback);
};

SVM.prototype.getKernelType = function() {
  var kernelTypes = _.map(KernelTypes, function(v, k){ return k;});
  var index = this._nodeSvm.getKernelType();
  return kernelTypes[index];
};

SVM.prototype.getSvmType = function() {
  var svmTypes = _.map(SvmTypes, function(v, k){ return k;});
  var index = this._nodeSvm.getSvmType();
  return svmTypes[index];
};



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

var meanNormalize = function(args){
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

  var sigmaInv = _.map(sigma, function(value){ return 1 / value;});
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

var readAndNormalizeProblemAsync = function(path, callback){
  readProblemAsync(path, function(problem){
    var result = meanNormalize({problem: problem});
    callback(result);
  });
};

exports.SvmTypes = SvmTypes;
exports.KernelTypes = KernelTypes;
exports.LinearKernel = LinearKernel;
exports.PolynomialKernel = PolynomialKernel;
exports.RadialBasisFunctionKernel = RadialBasisFunctionKernel;
exports.SigmoidKernel = SigmoidKernel;
exports.SVM = SVM;
exports.readProblemAsync = readProblemAsync;
exports.meanNormalize = meanNormalize;
exports.readAndNormalizeProblemAsync = readAndNormalizeProblemAsync;