'use strict';

var assert = require('assert'), 
    should = require('should'),
    _ = require('underscore'),
    async = require('async'),
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
  it('should return C=0.03125, gamma=0.5 on xor', function (done) {
    this.timeout(1000);
    var dataset = [];
    _.range(10).forEach(function(i){
      xorProblem.forEach(function (ex) {
        dataset.push(ex);
      });
    });
    
    var cValues = [0.03125, 0.125, 0.5, 2, 8],
        gValues = [8, 2, 0.5, 0.125, 0.03125];
    var options = {
      cValues: cValues,
      gValues: gValues,
      log: false
    }; 
    
    libsvm.findBestParameters(dataset, options, function(report) {
      cValues.should.containEql(report.C);
      gValues.should.containEql(report.gamma);
      report.accuracy.should.be.approximately(1, 0.01);
      report.fscore.should.be.approximately(1, 0.01);
      report.nbIterations.should.equal(cValues.length * gValues.length);
      done();
    });       
  });
});