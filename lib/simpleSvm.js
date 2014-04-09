'use strict';

var util = require('util'),
    events  = require('events'),
    _ = require('underscore'),
    numeric = require('numeric'),
    async = require('async'),
    nodesvm = require('./nodesvm');

var SimpleSvm = function (options) {
  events.EventEmitter.call(this);
  this._setOptions(options);
  this._mu = [];
  this._sigma = [];
  this._u = [];
  this._svm = null;
};
util.inherits(SimpleSvm, events.EventEmitter);

SimpleSvm.prototype._setOptions = function(options) {

  this._svmType = (typeof options.svmType) !== 'undefined' ? options.svmType : nodesvm.SvmTypes.C_SVC;

  this._kernelType = (typeof options.kernelType) !== 'undefined'? options.kernelType : nodesvm.KernelTypes.RBF;
  /*  DEGREE */
  if (this._kernelType === nodesvm.KernelTypes.POLY ){
    if (typeof options.degree !== 'undefined'){
      if (options.degree instanceof Array){
        this._degree = options.degree;
      }
      else {
        this._degree = [options.degree];
      }
    }
    else {
      this._degree = [2,3,4];
    } 
  }
  else {
    this._degree = []; // parameter degree used only for POLY kernel
  }
  /*  GAMMA */
  if (this._kernelType === nodesvm.KernelTypes.POLY ||
      this._kernelType === nodesvm.KernelTypes.RBF ||
      this._kernelType === nodesvm.KernelTypes.SIGMOID){
    if (typeof options.gamma !== 'undefined'){
      if (options.gamma instanceof Array){
        this._gamma = options.gamma;
      }
      else {
        this._gamma = [options.gamma];
      }
    }
    else {
      this._gamma = [0.03125, 0.125, 0.5, 2, 8];
    } 
  }
  else {
    this._gamma = []; 
  }
  /*  R */
  if (this._kernelType === nodesvm.KernelTypes.POLY  ||
      this._kernelType === nodesvm.KernelTypes.SIGMOID ){
    if (typeof options.r !== 'undefined'){
      if (options.r instanceof Array){
        this._r = options.r;
      }
      else {
        this._r = [options.r];
      }
    }
    else {
      this._r = [0.03125, 0.125, 0.5, 2, 8];
    } 
  }
  else {
    this._r = []; 
  }

  /*  C */
  if (this._svmType === nodesvm.SvmTypes.C_SVC ||
      this._svmType === nodesvm.SvmTypes.EPSILON_SVR ||
      this._svmType === nodesvm.SvmTypes.NU_SVR){
    if (typeof options.C !== 'undefined'){
      if (options.C instanceof Array){
        this._c = options.C;
      }
      else {
        this._c = [options.C];
      }
    }
    else {
      this._c = [0.03125, 0.125, 0.5, 2, 8];
    } 
  }
  else {
    this._c = []; 
  }

  /*  NU */
  if (this._svmType === nodesvm.SvmTypes.NU_SVC ||
      this._svmType === nodesvm.SvmTypes.ONE_CLASS ||
      this._svmType === nodesvm.SvmTypes.NU_SVR ){
    if (typeof options.nu !== 'undefined'){
      if (options.nu instanceof Array){
        this._nu = options.nu;
      }
      else {
        this._nu = [options.nu];
      }
    }
    else {
      this._nu = [0.03125, 0.125, 0.5, 0.75, 1];
    } 
  }
  else {
    this._nu = []; 
  }

  /*  EPSILON */
  if (this._svmType === nodesvm.SvmTypes.EPSILON_SVR){
    if (typeof options.epsilon !== 'undefined'){
      if (options.epsilon instanceof Array){
        this._epsilon = options.epsilon;
      }
      else {
        this._epsilon = [options.epsilon];
      }
    }
    else {
      this._epsilon = [0.03125, 0.125, 0.5, 2, 8];
    } 
  }
  else {
    this._epsilon = []; 
  }

  // training options
  this._normalize = (typeof options.normalize) === 'undefined' ? true : options.normalize;
  this._reduce = (typeof options.reduce) === 'undefined' ? true : options.reduce;
  this._retainedVariance = (typeof options.retainedVariance) === 'undefined' ? 0.99 : options.retainedVariance;
  this._eps = (typeof options.eps) === 'undefined' ? 1e-3 : options.eps;
  this._cacheSize = (typeof options.cacheSize) === 'undefined' ? 100 : options.cacheSize;
  this._probability = (typeof options.probability) === 'undefined' ? false : options.probability;
  this._nFold = (typeof options.nFold) === 'undefined' ? 4 : options.nFold;
};

SimpleSvm.prototype.train = function(dataset, callback) {   
  var ds = null, 
      self = this;
  if (typeof callback === 'function'){
    this.once('trained', callback);
  }
  if (this._normalize){
    var norm = nodesvm.meanNormalizeDataSet(dataset);
    ds = norm.dataset;
    this._mu = norm.mu;
    this._sigma = norm.sigma;
    this.emit('dataset-normalized', norm.mu, norm.sigma);
  }else {
    ds = dataset;
  }
  if (this._reduce){
    var red = nodesvm.reduceDatasetDimension(ds, this._retainedVariance);
    this._u = red.U;
    ds = red.dataset;
    this.emit('dataset-reduced', red.oldDimension, red.newDimension, red.retainedVariance, red.U);
  }
  var evalOptions = {
    svmType: self._svmType,
    kernelType: self._kernelType,
    fold: self._nFold,
    cValues: self._c,
    gValues: self._gamma,
    epsilonValues : self._epsilon,
    nuValues : self._nu,
    dValues : self._degree,
    rValues : self._r
  };
  nodesvm.findBestParameters(ds, evalOptions, function (evalReport) {
    var kernel = null;
    switch (self._kernelType){
    case nodesvm.KernelTypes.LINEAR : 
      kernel = new nodesvm.LinearKernel();
      break;
    case nodesvm.KernelTypes.RBF : 
      kernel = new nodesvm.RadialBasisFunctionKernel(evalReport.gamma);
      break;
    case nodesvm.KernelTypes.POLY : 
      kernel = new nodesvm.PolynomialKernel(evalReport.degree, evalReport.gamma, evalReport.r);
      break;
    case nodesvm.KernelTypes.SIGMOID : 
      kernel = new nodesvm.SigmoidKernel(evalReport.gamma, evalReport.r);
      break;
    default: 
      throw "Note supported kernel type";
    }

    var svm = new nodesvm.SVM({
      type: self._svmType,
      kernel: kernel,
      C: evalReport.C,
      nu: evalReport.nu,
      epsilon: evalReport.epsilon,

      eps: self._eps,
      cacheSize: self._cacheSize,
      probability: self._probability
    });
    self._svm = svm;
    svm.trainAsync(ds, function(){
      self.emit('trained', evalReport);
    });
  }, function(progressRate, remainingTime){
    // called during evaluation to report progress
    // remainingTime in ms
    self.emit('training-progressed', progressRate, remainingTime);
  });
};
SimpleSvm.prototype.evaluate = function(testset, callback) {
  var ds = null;
  if (this._normalize){
    ds = nodesvm.meanNormalizeDataSet({
      dataset:  testset, 
      mu: this._mu, 
      sigma: this._sigma
    }).dataset;
  }
  if (this.reduce){
    var self = this;
    ds = _.map(ds === null ? testset : ds, function(ex){
      var input = nodesvm.reduceInputDimension(ex[0], self._u);
      return [input, ex[1]];
    });
  }
  this._svm.evaluate(ds === null ? testset : ds, callback);
};
SimpleSvm.prototype._formatInputs = function(inputs){
  var x = null;
  if (this._normalize){
    x = nodesvm.meanNormalizeInput(inputs, this._mu, this._sigma);
  }
  if (this.reduce){
    x = nodesvm.reduceInputDimension(x === null ? inputs : x, this._u);
  }
  return x === null ? inputs : x;
};
SimpleSvm.prototype.predict = function(inputs) {
  var x = this._formatInputs(inputs);
  return this._svm.predict(x);
};

SimpleSvm.prototype.predictAsync = function(inputs, callback) {
  var x = this._formatInputs(inputs);
  return this._svm.predictAsync(x, callback);
};

SimpleSvm.prototype.predictProbabilities = function(inputs) {
  if (!this._probability){
    throw "Probabilities are disabled. Cannot predict probabilities";
  }
  var x = this._formatInputs(inputs);
  return this._svm.predictProbabilities(x);
};

SimpleSvm.prototype.predictProbabilitiesAsync = function(inputs, callback) {
  if (!this._probability){
    throw "Probabilities are disabled. Cannot predict probabilities";
  }
  var x = this._formatInputs(inputs);
  return this._svm.predictProbabilitiesAsync(x, callback);
};

SimpleSvm.prototype.getLabels = function() {
  return this._svm.labels;
};

var CSVC = function(options) {
  options['svmType'] = nodesvm.SvmTypes.C_SVC;
  SimpleSvm.call(this, options);
};
util.inherits(CSVC, SimpleSvm);

var NuSVC = function(options) {
  options['svmType'] = nodesvm.SvmTypes.NU_SVC;
  SimpleSvm.call(this, options);
};
util.inherits(NuSVC, SimpleSvm);

var EpsilonSVR = function(options) {
  options['svmType'] = nodesvm.SvmTypes.EPSILON_SVR;
  SimpleSvm.call(this, options);
};
util.inherits(EpsilonSVR, SimpleSvm);

var NuSVR = function(options) {
  options['svmType'] = nodesvm.SvmTypes.NU_SVR;
  SimpleSvm.call(this, options);
};
util.inherits(NuSVR, SimpleSvm);

exports.CSVC = CSVC;
exports.NuSVC = NuSVC;
exports.EpsilonSVR = EpsilonSVR;
exports.NuSVR = NuSVR;