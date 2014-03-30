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

/** 
  Linear Kernel
  K(xi , xj ) = transpose(xi) * xj
*/
var LinearKernel = function(){
  this.type = 'LINEAR';
};

/** 
  Polynomial Kernel
  K(xi , xj ) = Pow( gamma * transpose(xi) * xj + r , d )
  @degree: degree of the polynome
  @gamma: width parameter
*/
var PolynomialKernel = function(degree, gamma, r){
  this.type = 'POLY';
  this.degree = degree;
  this.gamma = gamma;
  this.r = r;
};

/** 
  RBF Kernel
  K(xi,yi) = exp( -gamma * || x - y ||² )
  @gamma: width parameter
*/
var RadialBasisFunctionKernel = function(gamma){
  this.type = 'RBF';
  this.gamma = gamma;
};

/** 
  Sigmoid Kernel
  K(xi , xj ) = tanh( gamma * transpose(xi) * xj + r)
  @gamma: width parameter
  @r: r parameter
*/
var SigmoidKernel = function(gamma, r){
  this.type = 'SIGMOID';
  this.gamma = gamma;
  this.r = r;
};

var SVM = function(args){
  events.EventEmitter.call(this);
  if (!args){
    args = {};
  }
  var svmTypes = _.map(SvmTypes, function(v, k){ return k;});
  this.svmType = svmTypes[args.type];
  this.kernel = args.kernel? args.kernel : {};

  this.C = args.C ?  args.C : 0.0;      // for C_SVC, EPSILON_SVR and NU_SVR
  this.cacheSize = args.cacheSize ? args.cacheSize: 100;// in MB 
  this.eps = args.eps ? args.eps : 1e-3; // stopping criteria
  this.nu = args.nu ? args.nu : 0.0; // for NU_SVC, ONE_CLASS, and NU_SVR
  this.p = args.p ? args.p : 0.0; // for EPSILON_SVR
  this.shrinking = args.shrinking ? args.shrinking : true; // use the shrinking heuristics
  this.probability = args.probability ? args.probability : true; // do probability estimates

  this._nodeSvm = new addon.NodeSvm();
};
util.inherits(SVM, events.EventEmitter);

SVM.prototype.train = function(problem, callback) {  
  var params = {
    type        : SvmTypes[this.svmType],
    kernel      : KernelTypes[this.kernel.type],
    degree      : this.kernel.degree ? this.kernel.degree : 0,
    gamma       : this.kernel.gamma ? this.kernel.gamma : 0,
    r           : this.kernel.r ? this.kernel.r : 0,
    cacheSize   : this.cacheSize,
    eps         : this.eps, 
    C           : this.C,
    // nrWeight    : this.weightLabel,
    // weightLabel : this.weightLabel ? 1 : 0,
    // weight      :  /* for C_SVC */
    nu          : this.nu,
    p           : this.p, 
    shrinking   : this.shrinking ? 1 : 0,
    probability : this.probability ? 1 : 0
  };
  var error = this._nodeSvm.setParameters(params);
  
  if (typeof callback === 'function' && error){
    callback(error);
    this.labels = [];
  }
  else{
    var self = this;
    this._nodeSvm.train(problem, function(){
      self.labels = self._nodeSvm.getLabels();
      callback();
    });
  }
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

var readProblem = function(path, callback){
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
      var node = {};
      node.x = numeric.rep([nbFeatures],0);
      for (var i = 1; i < elts.length ; i++){
        var indexValue = elts[i].split(":");
        var index = parseInt(indexValue[0], 10);
        var value = parseFloat(indexValue[1]);
        node.x[index - 1] = value;
      }
      node.y = parseFloat(elts[0]);
      result.push(node);
    });
    callback(result, nbFeatures);
  });
};

var meanNormalize = function(args, callback){
  var problem = args.problem;
  var X = _.pluck(problem, 'x');
  var n = numeric.dim(X)[0];
  if (n===0){
    callback(problem);
    return;
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
  for (var i = 0 ; i< m ; i++){
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
  for (var i2 = 0 ; i2< n ; i2++){
    var xNorm = numeric.mul(numeric.add(problem[i2].x, numeric.neg(mu)), sigmaInv);
    newProblem.push({
      x: xNorm,
      y: problem[i2].y
    });
  }
  callback(newProblem, mu, sigma);
};

exports.SvmTypes = SvmTypes;
exports.KernelTypes = KernelTypes;
exports.LinearKernel = LinearKernel;
exports.PolynomialKernel = PolynomialKernel;
exports.RadialBasisFunctionKernel = RadialBasisFunctionKernel;
exports.SigmoidKernel = SigmoidKernel;
exports.SVM = SVM;
exports.readProblem = readProblem;
exports.meanNormalize = meanNormalize;