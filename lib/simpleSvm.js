'use strict';

var util = require('util'),
    events  = require('events'),
    _ = require('underscore'),
    numeric = require('numeric'),
    async = require('async'),
    fs = require('fs'),
    nodesvm = require('./svm'),
    helpers = require('./helpers');

var SimpleSvm = function (options) {
  events.EventEmitter.call(this);
  if (typeof options.file !== 'undefined'){
    this._loadFromFile(options.file);
  }
  else if (typeof options.model !== 'undefined'){
    this._loadFromModel(options.model);
  }
  else{
    _.extend(options, {
      mu : [],
      sigma: [],
      u : []
    });
    this._setOptions(options);
    this._svm = null;
  }
};
util.inherits(SimpleSvm, events.EventEmitter);

SimpleSvm.prototype._loadFromFile = function(path) {
  if (!fs.existsSync(path)){
    throw '\'' + path + '\' not found';
  }
  var re = /\0/g;
  var content = fs.readFileSync(path, 'utf8').toString().replace(re, "");
  
  var args = JSON.parse(content);
  this._setOptions(args);

  // create a libsvm loading file 
  var libsvmModelContent = helpers.createLibsvmModelFileContent(args);
  

  // save the temporary file
  fs.writeFileSync(path + '.tmp', libsvmModelContent, 'utf8');
  this._svm = new nodesvm.SVM({file: path + '.tmp'});
  
  // remove the temp file
  fs.unlinkSync(path + '.tmp');

  // initialize PCA and normalization matrices
  this._mu = args.mu;
  this._sigma = args.sigma;
  this._u = args.u;
};
SimpleSvm.prototype._loadFromModel = function(model) {
  this._setOptions(model.params);
  this._svm = new nodesvm.SVM({model : model});
};

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

  /*  EPSILON, also called p */
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

  // features options
  this._normalize = (typeof options.normalize) === 'undefined' ? true : options.normalize;
  if (this._normalize){
    // check mu
    if ((typeof options.mu) === 'undefined'){
      throw 'mu must be defined if dataset normalized';
    }
    else if (! options.mu instanceof Array){
      throw 'mu must be an array';
    }
    else {
      this._mu =  options.mu;
    }
    // check sigma
    if ((typeof options.sigma) === 'undefined'){
      throw 'sigma must be defined if dataset normalized';
    }
    else if (! options.sigma instanceof Array){
      throw 'sigma must be an array';
    }
    else {
      this._sigma =  options.sigma;
    }
  }
  else {
    this._mu = [];
    this._sigma = [];
  }

  this._reduce = (typeof options.reduce) === 'undefined' ? true : options.reduce;
  this._retainedVariance = (typeof options.retainedVariance) === 'undefined' ? 0.99 : options.retainedVariance;
  if (this._reduce){
    // check mu
    if ((typeof options.u) === 'undefined'){
      throw 'u must be defined if dataset reduced';
    }
    else if (! options.u instanceof Array){
      throw 'u must be an array';
    }
    else {
      this._u =  options.u;
    }
  }
  else {
    this._u = [];
  }

  // training options
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
    var norm = helpers.meanNormalizeDataSet(dataset);
    ds = norm.dataset;
    this._mu = norm.mu;
    this._sigma = norm.sigma;
    this.emit('dataset-normalized', norm.mu, norm.sigma);
  }else {
    ds = dataset;
  }
  if (this._reduce){
    var red = helpers.reduceDatasetDimension(ds, this._retainedVariance);
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
  helpers.findBestParameters(ds, evalOptions, function (evalReport) {
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

SimpleSvm.prototype.trainFromFile = function(path, callback) {
  var self = this;
  if (typeof callback === 'function'){
    this.once('trained', callback);
  }
  helpers.readDatasetAsync(path, function(ds){
    self.train(ds);
  });
};

SimpleSvm.prototype.evaluate = function(testset, callback) {
  var ds = null;
  if (this._normalize){
    ds = helpers.meanNormalizeDataSet({
      dataset:  testset, 
      mu: this._mu, 
      sigma: this._sigma
    }).dataset;
  }
  if (this.reduce){
    var self = this;
    ds = _.map(ds === null ? testset : ds, function(ex){
      var input = helpers.reduceInputDimension(ex[0], self._u);
      return [input, ex[1]];
    });
  }
  this._svm.evaluate(ds === null ? testset : ds, callback);
};
SimpleSvm.prototype._formatInputs = function(inputs){
  var x = null;
  if (this._normalize){
    x = helpers.meanNormalizeInput(inputs, this._mu, this._sigma);
  }
  if (this._reduce){
    x = helpers.reduceInputDimension(x === null ? inputs : x, this._u);
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

SimpleSvm.prototype.saveToFile = function(path) {
  var model=  this.getModel();
  fs.writeFileSync(path, JSON.stringify(model, null, 2));
};

SimpleSvm.prototype.getModel = function(){
  var baseModel  = this._svm.getModel();
  _.extend(baseModel.params, {
    normalize: this._normalize,
    mu: this._mu,
    sigma : this._sigma,
    reduce : this._reduce,
    retainedVariance : this._retainedVariance,
    u : this._u
  });
  return baseModel;
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

exports.SimpleSvm = SimpleSvm;
exports.CSVC = CSVC;
exports.NuSVC = NuSVC;
exports.EpsilonSVR = EpsilonSVR;
exports.NuSVR = NuSVR;