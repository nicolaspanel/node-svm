'use strict';

var assert = require('assert'), 
    should = require('should'),
    _ = require('underscore'),
    async = require('async'),
    numeric = require('numeric'),
    libsvm = require('../lib/nodesvm');

var xorProblem = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0]
];
var xorNormProblem = [
  [[-1, -1], 0],
  [[-1,  1], 1],
  [[ 1, -1], 1],
  [[ 1,  1], 0]
];

describe('#readProblemAsync', function(){  
  it('should be able to read the xor problem', function (done) {
    libsvm.readProblemAsync('./examples/datasets/xor.ds', function(problem, nbFeature){
      nbFeature.should.equal(2);
      problem.length.should.equal(4);
      problem.should.eql(xorProblem);
      done();
    });
  });
  it('should be able to read the svmguide problem in less than 200ms', function (done) {
    this.timeout(200);
    libsvm.readProblemAsync('./examples/datasets/svmguide1.ds', function(problem, nbFeature){
      nbFeature.should.equal(4);
      problem.length.should.equal(3089);
      done();
    });
  });
});

describe('#meanNormalizeDataSet', function(){  
  
  it('should be able to Mean Normalize the xor problem', function () {
    var result = libsvm.meanNormalizeDataSet({dataset: xorProblem});
    result.mu.should.eql([0.5, 0.5]);
    result.sigma.should.eql([0.5,0.5]);
    result.dataset.should.eql(xorNormProblem);
  });

  it('should be able to Mean Normalize an already normalized problem', function() {
    var result = libsvm.meanNormalizeDataSet({dataset: xorNormProblem});
    result.dataset.should.eql(xorNormProblem);
  });
  
  it('should be able to Mean Normalize the xor problem with custom mu and sigma', function () {
    var result = libsvm.meanNormalizeDataSet({dataset: xorProblem, mu: [0, 0], sigma: [1, 1]});
    result.dataset.should.eql(xorProblem); // no changes
  });
});

describe('#meanNormalizeInput', function(){  
  
  it('should be able to Mean Normalize the xor problem', function () {
    libsvm.meanNormalizeInput([0, 0], [0.5, 0.5], [0.5,0.5]).should.eql([-1, -1]);
  });

});

describe('#readAndNormalizeDatasetAsync', function(){  
  it('should be able to read and mean normalize the xor problem', function (done) {
    libsvm.readAndNormalizeDatasetAsync('./examples/datasets/xor.ds', function(result){
      result.mu.should.eql([0.5, 0.5]);
      result.sigma.should.eql([0.5,0.5]);
      result.dataset.should.eql(xorNormProblem);
      done();
    });      
  });
});

describe('#performNFoldCrossValidation', function(){  
  it('should predict an 100% accuracy against XOR', function (done) {
    var dataset = [], svm = null;
    _.range(50).forEach(function(i){
      xorProblem.forEach(function (ex) {
        dataset.push(ex);
      });
    });
    
    svm = new libsvm.SVM({
      type: libsvm.SvmTypes.C_SVC,
      kernel: new libsvm.RadialBasisFunctionKernel(0.5),
      C: 1
    });
    var kfold = 4;
    
    libsvm.performNFoldCrossValidation(svm, dataset, kfold, function(report){
      report.accuracy.should.eql(1);
      done();
    });      
  });
});

describe('#evaluateSvm', function(){  
  it('should predict an 100% accuracy on XOR', function (done) {
    var dataset = xorProblem, 
        testset  = xorProblem,
        svm = null;
    
    svm = new libsvm.SVM({
      type: libsvm.SvmTypes.C_SVC,
      kernel: new libsvm.RadialBasisFunctionKernel(0.5),
      C: 1
    });
    
    svm.train(dataset);
    
    libsvm.evaluateSvm(svm, testset, function(report){
      report.accuracy.should.eql(1);
      done();
    });      
  });
});

describe('#findBestParameters', function(done){  
  var dataset = null;
  beforeEach(function() {
    dataset = [];
    _.range(10).forEach(function(i){
      xorProblem.forEach(function (ex) {
        dataset.push(ex);
      });
    });
  });
  
  it('should work on xor dataset with C-SVC and RBF kernel', function (done) {
    var options = {
      // svmType : libsvm.SvmTypes.C_SVC,     // (default value)
      // kernelType : libsvm.KernelTypes.RBF, // (default value)
      cValues: [0.03125, 0.125, 0.5, 2, 8],
      gValues: [8, 2, 0.5, 0.125, 0.03125]
    }; 
    
    libsvm.findBestParameters(dataset, options, function(report) {
      options.cValues.should.containEql(report.C);
      options.gValues.should.containEql(report.gamma);
      report.accuracy.should.be.approximately(1, 0.01);
      report.fscore.should.be.approximately(1, 0.01);
      report.nbIterations.should.equal(25);
      done();
    });       
  });
  
  it('should work on xor dataset with EPSILON_SVR and RBF kernel', function (done) {
    var options = {
      svmType : libsvm.SvmTypes.EPSILON_SVR,
      //kernelType : libsvm.KernelTypes.RBF, // (default value)
      cValues: [0.03125, 0.125, 0.5, 2, 8],
      gValues: [8, 2, 0.5, 0.125, 0.03125],
      epsilonValues: [8, 2, 0.5, 0.125, 0.03125]
    }; 
    
    libsvm.findBestParameters(dataset, options, function(report) {
      options.cValues.should.containEql(report.C);
      options.epsilonValues.should.containEql(report.epsilon);
      report.mse.should.be.approximately(0, 1e-3);
      report.nbIterations.should.equal(125);
      done();
    });       
  });

  it('should work on xor dataset with NU_SVR and SIGMOID kernel', function (done) {
    var completion = 0;
    var options = {
      svmType : libsvm.SvmTypes.NU_SVR,
      kernelType : libsvm.KernelTypes.SIGMOID,
      gValues: [8, 2, 0.5, 0.125, 0.03125], // for sigmoid kernel
      rValues: [0], // for sigmoid kernel
      nuValues: [0, 0.25, 0.5, 0.75, 1], // for NU_SVR
      cValues: [8] // for NU_SVR
    }; 
    
    libsvm.findBestParameters(dataset, options, function(report) {
      options.gValues.should.containEql(report.gamma);
      options.rValues.should.containEql(report.r);
      options.nuValues.should.containEql(report.nu);
      report.mse.should.be.approximately(0.25, 1e-3);
      report.nbIterations.should.equal(25);
      completion.should.equal(1);
      done();
    }, function(progress){
      completion = progress;
    });       
  });
});

describe('#findAllPossibleCombinaisons', function() {
  var A = [0, 1, 2],
      B = [0, 1],
      C = [];

  it('should have a size of 6x2 for A u B', function () {
    numeric.dim(libsvm.findAllPossibleCombinaisons([A, B])).should.eql([6,2]);
  });

  it('should return all possible combinaisons for A u B', function () {
    var expected = [
      [0, 0],
      [1, 0],
      [2, 0],
      [0, 1],
      [1, 1],
      [2, 1]
    ];
    libsvm.findAllPossibleCombinaisons([A, B]).should.eql(expected);
  });

  it('should have a size of 6x3 for A u B u C', function () {
    numeric.dim(libsvm.findAllPossibleCombinaisons([A, B, C])).should.eql([6,3]);
  });

  it('should return all possible combinaisons for A u C u B u C', function () {
    var expected = [
      [0, 0, 0, 0],
      [1, 0, 0, 0],
      [2, 0, 0, 0],
      [0, 0, 1, 0],
      [1, 0, 1, 0],
      [2, 0, 1, 0]
    ];
    var result = libsvm.findAllPossibleCombinaisons([A, C, B, C]);
    result.should.eql(expected);
  });
});