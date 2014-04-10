'use strict';

var assert = require('assert'), 
    should = require('should'),
    _ = require('underscore'),
    async = require('async'),
    fs = require('fs'),
    nodesvm = require('../lib/nodesvm');

var xorProblem = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
];
var xorFileName = './examples/datasets/xor.ds';
describe('C-SVC', function(){
  var svm = null;
  beforeEach(function(){
    var options = {       
    //   // kernels parameters
    //   kernelType: nodesvm.KernelTypes.RBF,  
    //   degree: [2,3,4],
    //   gamma: [0.03125, 0.125, 0.5, 2, 8],
    //   r: [0.125, 0.5, 2, 8],
    //   C: [0.03125, 0.125, 0.5, 2, 8],      // cost for C_SVC, EPSILON_SVR and NU_SVR
    //   nu: [0.03125, 0.125, 0.5, 0.75, 1], // for NU_SVC, ONE_CLASS SVM, and NU_SVR
    //   epsilon : [0.03125, 0.125, 0.5, 2, 8], // set the epsilon in loss function

    //   // training options
    //   nFold: 4,
    //   normalize: true, 
    //   reduce: true, // use PCA to reduce dataset size
    //   retainedVariance: 0.99,
    //   eps: 1e-3, // stopping criteria 
    //   cacheSize: 100,     // cache siez in MB        
    //   probability : false // false by default
    }; // default
    svm = new nodesvm.CSVC(options);
  });
  it('should use C_SVC type', function() {
    svm._svmType.should.equal(nodesvm.SvmTypes.C_SVC);
  });

  it('should use RBF kernel', function() {
    svm._kernelType.should.equal(nodesvm.KernelTypes.RBF);
  });

  it('should have no degree parameter', function() {
    svm._degree.length.should.equal(0);
  });

  it('should have gamma set to default', function() {
    svm._gamma.should.eql([0.03125, 0.125, 0.5, 2, 8]);
  });

  it('should have no r parameter', function() {
    svm._r.length.should.equal(0);
  });

  it('should have C set to default', function() {
    svm._c.should.eql([0.03125, 0.125, 0.5, 2, 8]);
  });

  it('should have no nu parameter', function() {
    svm._nu.length.should.equal(0);
  });

  it('should have no epsilon parameter', function() {
    svm._epsilon.length.should.equal(0);
  });

  it('should use normalization', function() {
    svm._normalize.should.be.true;
  });

  it('should use PCA', function() {
    svm._reduce.should.be.true;
  });

  it('should have retainedVariance set to 0.99', function() {
    svm._retainedVariance.should.equal(0.99);
  });

  it('should have stopping criteria set to 1e-3', function() {
    svm._eps.should.equal(1e-3);
  });
  
  it('should have cache size set to 100', function() {
    svm._cacheSize.should.equal(100);
  });
  
  it('should NOT use probabilities', function() {
    svm._probability.should.be.false;
  });

  it('should have nFold set to 4', function() {
    svm._nFold.should.equal(4);
  });
  
  it('should emit `dataset-normalized` event during evaluation', function(done){
    svm.once('dataset-normalized', function () {
      done();
    });
    svm.train(xorProblem);
  });

  it('should emit `dataset-reduced` event during evaluation', function(done){
    svm.once('dataset-reduced', function (oldDim, newDim, retainedVar) {
      newDim.should.equal(2);
      retainedVar.should.equal(1);
      done();
    });
    svm.train(xorProblem);
  });

  it('should emit `training-progressed` event during evaluation', function(done){
    svm.once('training-progressed', function (progressRate, remainingTime) {
      done();
    });
    svm.train(xorProblem);
  });

  it('should emit `trained` event once trained', function(done){
    svm.once('trained', function () {
      done();
    });
    svm.train(xorProblem);
  });

  it('should have a trained callback', function(done){
    svm.train(xorProblem, function () {
      done();
    });
  });

  it('can be train from file', function(done){
    svm.trainFromFile(xorFileName, function (report) {
      report.accuracy.should.equal(1);
      done();
    });
  });

  describe('once trained on xor problem', function(){
    beforeEach(function(done){
      svm.train(xorProblem, function() {
        done();
      });
    });
    it('can evaluate itself against a testset', function(done){
      svm.evaluate(xorProblem, function (report) {
        report.accuracy.should.equal(1);
        done();
      });     
    });
    it('should perform very well on the training set (100% accuracy)', function(){
      xorProblem.forEach(function(ex){
        var prediction = svm.predict(ex[0]);
        prediction.should.equal(ex[1]);  //  means that y E {0;1}
      });
    });
    it('should be able to predict Async', function(done){
      async.each(xorProblem, function(ex, callback) {
        svm.predictAsync(ex[0], function(prediction){
          prediction.should.equal(ex[1]);
          callback();
        });
      }, function(err){ done(); });
    });
    
    it('should throw an error if try to predict probabilities', function(){
      var testFunc = function(){
        svm.predictProbabilities(xorProblem[0][0]);
      };
      testFunc.should.throw();
    });
    it('should throw an error if try to predict probabilities', function(){
      var testFunc = function(){
        svm.predictProbabilitiesAsync(xorProblem[0][0]);
      };
      testFunc.should.throw();
    });
    it('can be save in a file', function(){
      var fileName = './test/models/simple_csvc.model';
      svm.saveToFile(fileName);
      fs.existsSync(fileName).should.be.true;
      fs.unlinkSync(fileName);
    });
  });
});

describe('NU-SVC', function(){
  var svm = null;
  beforeEach(function(){
    var options = {       
      kernelType: nodesvm.KernelTypes.POLY,
      gamma: 2,
      r: 1,
      nu: 0.1,
      nFold: 3,
      normalize: false,
      reduce: false,
      eps: 1e-5,
      cacheSize: 200,
      probability: true
    }; // default
    svm = new nodesvm.NuSVC(options);
  });
  it('should use NU_SVC type', function() {
    svm._svmType.should.equal(nodesvm.SvmTypes.NU_SVC);
  });

  it('should use RBF kernel', function() {
    svm._kernelType.should.equal(nodesvm.KernelTypes.POLY);
  });

  it('should have degree set to default', function() {
    svm._degree.should.eql([2,3,4]);
  });

  it('should have one parameter for gamma', function() {
    svm._gamma.should.eql([2]);
  });

  it('should have one parameter for r', function() {
    svm._r.should.eql([1]);
  });

  it('should have no parameter for C', function() {
    svm._c.length.should.equal(0);
  });
  it('should have one parameter for nu', function() {
    svm._nu.should.eql([0.1]);
  });

  it('should have nFold set to 3', function() {
    svm._nFold.should.equal(3);
  });
  
  it('should NOT use normalization', function() {
    svm._normalize.should.be.false;
  });
  it('should NOT use PCA', function() {
    svm._reduce.should.be.false;
  });
  it('should have stopping criteria set to 1e-5', function() {
    svm._eps.should.equal(1e-5);
  });
  it('should have cache size set to 200', function() {
    svm._cacheSize.should.equal(200);
  });
  
  it('should use probabilities', function() {
    svm._probability.should.be.true;
  });

  describe('once trained with xor', function () {
    beforeEach(function(done){
      svm.train(xorProblem, function() {
        done();
      });
    });
    it('should be able to predict probabilities', function(){
      xorProblem.forEach(function(ex){
        var probs = svm.predictProbabilities(ex[0]);
        var sum = 0;
        svm.getLabels().forEach(function (classLabel) {
          sum += probs[classLabel];
        });
        sum.should.be.approximately(1, 1e-5);
      });
    });

    it('should be able to predict probabilities async', function(done){
      async.each(xorProblem, function(ex, cb){
        svm.predictProbabilitiesAsync(ex[0], function(probabilities){
          var sum = 0;
          svm.getLabels().forEach(function (classLabel) {
            sum += probabilities[classLabel];
          });
          sum.should.be.approximately(1, 1e-5);
          cb();
        });
      }, function(err){ done(); });
    });
    it('can be saved into a file', function(){
      var fileName = './examples/models/simple_nusvc.model';
      svm.saveToFile(fileName);
      fs.existsSync(fileName).should.be.true;
      fs.unlinkSync(fileName);
    });
  });
});

describe('EPSILON-SVR', function(){
  var svm = null;
  beforeEach(function(){
    var options = {       
      kernelType: nodesvm.KernelTypes.LINEAR,
      C: 2  
    }; // default
    svm = new nodesvm.EpsilonSVR(options);
  });
  it('should use EPSILON_SVR type', function() {
    svm._svmType.should.equal(nodesvm.SvmTypes.EPSILON_SVR);
  });

  it('should use LINEAR kernel', function() {
    svm._kernelType.should.equal(nodesvm.KernelTypes.LINEAR);
  });

  it('should have no parameter for gamma', function() {
    svm._gamma.length.should.equal(0);
  });

  it('should have one parameter for C', function() {
    svm._c.should.eql([2]);
  });

  it('should have epsilon parameter set to default', function() {
    svm._epsilon.should.eql([0.03125, 0.125, 0.5, 2, 8]);
  });

  describe('once trained with xor', function () {
    beforeEach(function(done){
      svm.train(xorProblem, function() {
        done();
      });
    });
    
    it('can be saved into a file', function(){
      var fileName = './test/models/simple_epsilonsvr.model';
      svm.saveToFile(fileName);
      fs.existsSync(fileName).should.be.true;
      fs.unlinkSync(fileName);
    });
  });
});

describe('NU-SVR', function(){
  var svm = null;
  beforeEach(function(){
    var options = {}; // default
    svm = new nodesvm.NuSVR(options);
  });
  it('should use NU_SVR type', function() {
    svm._svmType.should.equal(nodesvm.SvmTypes.NU_SVR);
  });
  it('should have nu set to default', function() {
    svm._nu.should.eql([0.03125, 0.125, 0.5, 0.75, 1]);
  });
  describe('once trained with xor', function () {
    beforeEach(function(done){
      svm.train(xorProblem, function() {
        done();
      });
    });
    
    it('can be saved into a file', function(){
      var fileName = './test/models/simple_nusvr.model';
      svm.saveToFile(fileName);
      fs.existsSync(fileName).should.be.true;
      fs.unlinkSync(fileName);
    });
  });
});


describe('Basic XOR', function(){
  var fileName = './test/models/basic-xor.model';
  var svm = null;
  beforeEach(function(done){
    var tempSvm = new nodesvm.CSVC({
      kernelType: nodesvm.KernelTypes.RBF,
      gamma: 0.5,
      C: 1,
      normalize: false,
      reduce: false
    });
    tempSvm.train(xorProblem, function () {
      tempSvm.saveToFile(fileName);
      svm = nodesvm.loadSvmFromFile(fileName);
      done();
    });
  });
  afterEach(function () {
    if (fs.existsSync(fileName)){
      fs.unlinkSync(fileName);
    }
  });
  it('should be an instance of SimpleSvm', function () {
    svm.should.be.an.instanceOf(nodesvm.SimpleSvm);
  });   
  it('can be used by other svm to predict XOR', function () {
    xorProblem.forEach(function(ex){
      var prediction = svm.predict(ex[0]);
      prediction.should.equal(ex[1]);  //  means that y E {0;1}
    });
  });
  it('should have remove the *.tmp file', function () {
    fs.existsSync(fileName + '.tmp').should.be.false;
  });
  it('should use C_SVC type', function() {
    svm._svmType.should.equal(nodesvm.SvmTypes.C_SVC);
  });
  it('should use RBF kernel', function() {
    svm._kernelType.should.equal(nodesvm.KernelTypes.RBF);
  });

  it('should have gamma set to 0.5', function() {
    svm._gamma.should.eql([0.5]);
  });
  
  it('should NOT use normalization', function() {
    svm._normalize.should.be.false;
  });

  it('should NOT use PCA', function() {
    svm._reduce.should.be.false;
  });
      
});

describe('Advanced XOR load from file', function(){
  var fileName = './examples/models/advanced-xor.model';
  var svm = null;
  beforeEach(function(done){
    var tempSvm = new nodesvm.CSVC({
      kernelType: nodesvm.KernelTypes.RBF,
      gamma: 0.5,
      C: 1,
      normalize: true,
      reduce: true
    });
    tempSvm.train(xorProblem, function () {
      tempSvm.saveToFile(fileName);
      svm = nodesvm.loadSvmFromFile(fileName);
      done();
    });
  });
  afterEach(function () {
    if (fs.existsSync(fileName)){
      fs.unlinkSync(fileName);
    }
  });
      
  it('can be used by other svm to predict XOR', function () {
    xorProblem.forEach(function(ex){
      var prediction = svm.predict(ex[0]);
      prediction.should.equal(ex[1]);  //  means that y E {0;1}
    });
  });
  
  it('should use normalization', function() {
    svm._normalize.should.be.true;
  });
  it('should have mu set to [0.5,0.5]', function() {
    svm._mu.should.eql([0.5,0.5]);
  });
  it('should have sigma set to [0.5,0.5]', function() {
    svm._sigma.should.eql([0.5,0.5]);
  });

  it('should use PCA', function() {
    svm._reduce.should.be.true;
  });
  
  it('should have u equal to Identity Matrix', function() {
    svm._u.should.eql([[-1,0],[0,-1]]);
  });
});