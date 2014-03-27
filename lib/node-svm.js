'use strict';

var util = require('util'),
    events  = require('events'),
    _ = require('underscore'),
    addon = require('../build/Release/addon');

var SvmTypes = {
  C_SVC: 0,
  NU_SVC: 1,
  ONE_CLASS: 2,
  EPSILON_SVR : 3,
  NU_SVR: 4
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
  this.kernel = args.kernel;

  this.C = args.C ?  args.C : 0.0;      // for C_SVC, EPSILON_SVR and NU_SVR
  this.cacheSize = args.cacheSize ? args.cacheSize: 100;// in MB 
  this.eps = args.eps ? args.eps : 1e-3; // stopping criteria
  this.nu = args.nu ? args.nu : 0.0; // for NU_SVC, ONE_CLASS, and NU_SVR
  this.p = args.p ? args.p : 0.0; // for EPSILON_SVR
  this.shrinking = args.shrinking ? args.shrinking : true; // use the shrinking heuristics
  this.probability = args.probability ? args.probability : true; // do probability estimates
};
util.inherits(SVM, events.EventEmitter);

SVM.prototype.train = function(problem, callback) {
  if (typeof callback === 'function'){
    this.once('trained', callback);
  }
  var self = this;
  var params = {
    type        : SvmTypes[this.svmType],
    kernel      : KernelTypes[this.kernel],
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
  addon.svmTrain(problem, params, function () {
    self.emit('trained');
  });
};

exports.SvmTypes = SvmTypes;
exports.KernelTypes = KernelTypes;
exports.LinearKernel = LinearKernel;
exports.PolynomialKernel = PolynomialKernel;
exports.RadialBasisFunctionKernel = RadialBasisFunctionKernel;
exports.SigmoidKernel = SigmoidKernel;
exports.SVM = SVM;