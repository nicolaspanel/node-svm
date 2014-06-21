'use strict';

var util = require('util'),
    events  = require('events'),
    _ = require('underscore'),
    fs = require('fs'),
    addon = require('../build/Release/addon'),
    classificator = require('./evaluators');

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
  if (!args){
    args = {};
  }
  
  this._nodeSvm = new addon.NodeSvm();
  if (args.file){
    if (!fs.existsSync(args.file)){
      throw '\'' + args.file + '\' not found';
    }
    this._nodeSvm.loadFromFile(args.file);
  }
  else if (args.model){
    this._nodeSvm.loadFromModel(args.model);
  }
  else{
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
    else {
      C = args.C;
    }

    var params = {
      svmType: svmType, // default: 0 -- C_SVC
      kernelType: kernel.kernelType,
      degree: kernel.degree,
      gamma: kernel.gamma,
      r: kernel.r,
      C: C,      // cost for C_SVC, EPSILON_SVR and NU_SVR
      nu: args.nu ? args.nu : 0.5, // for NU_SVC, ONE_CLASS SVM, and NU_SVR
      epsilon : args.epsilon ? args.epsilon : 0.1, // set the epsilon in loss function of epsilon-SVR (default 0.1)
      eps: args.eps ? args.eps : 1e-3, // stopping criteria 
      cacheSize: args.cacheSize ? args.cacheSize: 100,                 // in MB 
      shrinking   : 1, // always use the shrinking heuristics
      probability : args.probability ? 1 : 0 // false by default
    };
    
    var error = this._nodeSvm.setParameters(params);
    if (error){
      throw "Invalid parameter. Err: " + error;
    } 
  }
};

SVM.prototype.train = function(problem) { 
  this._nodeSvm.train(problem);
  var svmType = this.getSvmType();
  if (svmType === 'C_SVC' || svmType === 'NU_SVC'){
    this.labels = this._nodeSvm.getLabels();
  }
  else{
    this.labels = []; // no labels in regression
  }
};

SVM.prototype.trainAsync = function(problem, callback) {  
  var self = this;
  this._nodeSvm.trainAsync(problem, function () {
    var svmType = self.getSvmType();
    if (svmType === 'C_SVC' || svmType === 'NU_SVC'){
      self.labels = self._nodeSvm.getLabels();
    }
    else{
      self.labels = []; // no labels in regression
    }
    callback();
  });
};

SVM.prototype.predict = function(data) {  
  return this._nodeSvm.predict(data);
};
SVM.prototype.predictAsync = function(data, callback) {  
  return this._nodeSvm.predictAsync(data, callback);
};

/**
NOTE : To use this function, make sure to active probabilities during the SVM initialization. Example : 
  svc = new svm.SVM({
    type: svm.SvmTypes.C_SVC,
    kernel: new svm.RadialBasisFunctionKernel(0.5),
    C: 1,
    probability: true
  });

WARNING : Seems not to work very well.
see : http://stats.stackexchange.com/questions/64403/libsvm-probability-estimates-in-multi-class-problems
*/
SVM.prototype.predictProbabilities = function(data) {
  var probs = this._nodeSvm.predictProbabilities(data);
  var result = {};
  for (var i = 0; i < probs.length ; i++){
    result[this.labels[i]] = probs[i];
  }
  return result;
};

SVM.prototype.predictProbabilitiesAsync = function(data, callback) {
  var self = this;
  this._nodeSvm.predictProbabilitiesAsync(data, function(probs){
    var result = {};
    for (var i = 0; i < probs.length ; i++){
      result[self.labels[i]] = probs[i];
    }
    callback(result);
  });
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

SVM.prototype.isTrained = function() {
  return this._nodeSvm.isTrained();
};

SVM.prototype.saveToFile = function(path) {
  this._nodeSvm.saveToFile(path);
};

SVM.prototype.getModel = function() {
  return this._nodeSvm.getModel();
};

SVM.prototype.evaluate = function(testset, callback) {
  var type = this.getSvmType(), 
      evaluator = null;
  if (type === 'NU_SVC' || type === 'C_SVC'){
    evaluator = new classificator.ClassificationEvaluator(this);
  }
  else if (type === 'EPSILON_SVR' || type === 'NU_SVR'){
    evaluator = new classificator.RegressionEvaluator(this);
  }
  else{
    throw 'Not supported for yet';
  }
  evaluator.evaluate(testset, callback);
};

SVM.prototype.performNFoldCrossValidation = function(dataset, nfold, done) {
  var type = this.getSvmType(), 
      evaluator = null;
  if (type === 'NU_SVC' || type === 'C_SVC'){
    evaluator = new classificator.ClassificationEvaluator(this);
  }
  else if (type === 'EPSILON_SVR' || type === 'NU_SVR'){
    evaluator = new classificator.RegressionEvaluator(this);
  }
  else{
    throw 'Not supported for yet';
  }

  if (nfold > 1 && nfold < dataset.length){
    evaluator.performNFoldCrossValidation(nfold, dataset, done);
  }
  else {
    if (this.isTrained()){
      evaluator.evaluate(dataset, done);
    }
    else {
      this.trainAsync(dataset, function(){
        evaluator.evaluate(dataset, done);
      });
    }
  }
};

exports.SvmTypes = SvmTypes;
exports.KernelTypes = KernelTypes;
exports.LinearKernel = LinearKernel;
exports.PolynomialKernel = PolynomialKernel;
exports.RadialBasisFunctionKernel = RadialBasisFunctionKernel;
exports.SigmoidKernel = SigmoidKernel;
exports.SVM = SVM;
