'use strict';

var assert = require('assert'), 
    should = require('should'),
    _ = require('underscore'),
    async = require('async'),
    numeric = require('numeric'),
    nodesvm = require('../lib/nodesvm');

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

describe('#readDatasetAsync', function(){  
  it('should be able to read the xor problem', function (done) {
    nodesvm.readDatasetAsync('./examples/datasets/xor.ds', function(problem, nbFeature){
      nbFeature.should.equal(2);
      problem.length.should.equal(4);
      problem.should.eql(xorProblem);
      done();
    });
  });
  it('should be able to read the svmguide problem in less than 200ms', function (done) {
    this.timeout(200);
    nodesvm.readDatasetAsync('./examples/datasets/svmguide1.ds', function(problem, nbFeature){
      nbFeature.should.equal(4);
      problem.length.should.equal(3089);
      done();
    });
  });
});

describe('#meanNormalizeDataSet', function(){  
  
  it('should be able to Mean Normalize the xor problem', function () {
    var result = nodesvm.meanNormalizeDataSet({dataset: xorProblem});
    result.mu.should.eql([0.5, 0.5]);
    result.sigma.should.eql([0.5,0.5]);
    result.dataset.should.eql(xorNormProblem);
  });

  it('should be able to Mean Normalize an already normalized problem', function() {
    var result = nodesvm.meanNormalizeDataSet({dataset: xorNormProblem});
    result.dataset.should.eql(xorNormProblem);
  });
  
  it('should be able to Mean Normalize the xor problem with custom mu and sigma', function () {
    var result = nodesvm.meanNormalizeDataSet({dataset: xorProblem, mu: [0, 0], sigma: [1, 1]});
    result.dataset.should.eql(xorProblem); // no changes
  });
});

describe('#meanNormalizeInput', function(){  
  
  it('should be able to Mean Normalize the xor problem', function () {
    nodesvm.meanNormalizeInput([0, 0], [0.5, 0.5], [0.5,0.5]).should.eql([-1, -1]);
  });

});

describe('#readAndNormalizeDatasetAsync', function(){  
  it('should be able to read and mean normalize the xor problem', function (done) {
    nodesvm.readAndNormalizeDatasetAsync('./examples/datasets/xor.ds', function(result){
      result.mu.should.eql([0.5, 0.5]);
      result.sigma.should.eql([0.5,0.5]);
      result.dataset.should.eql(xorNormProblem);
      done();
    });      
  });
});

describe('#performNFoldCrossValidation', function(){  
  it('should predict an 100% accuracy against XOR', function (done) {
    var dataset = [], svc = null;
    _.range(50).forEach(function(i){
      xorProblem.forEach(function (ex) {
        dataset.push(ex);
      });
    });
    
    svc = new nodesvm.SVM({
      type: nodesvm.SvmTypes.C_SVC,
      kernel: new nodesvm.RadialBasisFunctionKernel(0.5),
      C: 1
    });
    var kfold = 4;
    
    nodesvm.performNFoldCrossValidation(svc, dataset, kfold, function(report){
      report.accuracy.should.eql(1);
      done();
    });      
  });
});

describe('#evaluateSvm', function(){  
  it('should predict an 100% accuracy on XOR', function (done) {
    var dataset = xorProblem, 
        testset  = xorProblem,
        svc = null;
    
    svc = new nodesvm.SVM({
      type: nodesvm.SvmTypes.C_SVC,
      kernel: new nodesvm.RadialBasisFunctionKernel(0.5),
      C: 1
    });
    
    svc.train(dataset);
    
    nodesvm.evaluateSvm(svc, testset, function(report){
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
      // svmType : nodesvm.SvmTypes.C_SVC,     // (default value)
      // kernelType : nodesvm.KernelTypes.RBF, // (default value)
      cValues: [0.03125, 0.125, 0.5, 2, 8],
      gValues: [8, 2, 0.5, 0.125, 0.03125]
    }; 
    
    nodesvm.findBestParameters(dataset, options, function(report) {
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
      svmType : nodesvm.SvmTypes.EPSILON_SVR,
      //kernelType : nodesvm.KernelTypes.RBF, // (default value)
      cValues: [0.03125, 0.125, 0.5, 2, 8],
      gValues: [8, 2, 0.5, 0.125, 0.03125],
      epsilonValues: [8, 2, 0.5, 0.125, 0.03125]
    }; 
    
    nodesvm.findBestParameters(dataset, options, function(report) {
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
      svmType : nodesvm.SvmTypes.NU_SVR,
      kernelType : nodesvm.KernelTypes.SIGMOID,
      gValues: [8, 2, 0.5, 0.125, 0.03125], // for sigmoid kernel
      rValues: [0], // for sigmoid kernel
      nuValues: [0, 0.25, 0.5, 0.75, 1], // for NU_SVR
      cValues: [8] // for NU_SVR
    }; 
    
    nodesvm.findBestParameters(dataset, options, function(report) {
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
    numeric.dim(nodesvm.findAllPossibleCombinaisons([A, B])).should.eql([6,2]);
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
    nodesvm.findAllPossibleCombinaisons([A, B]).should.eql(expected);
  });

  it('should have a size of 6x3 for A u B u C', function () {
    numeric.dim(nodesvm.findAllPossibleCombinaisons([A, B, C])).should.eql([6,3]);
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
    var result = nodesvm.findAllPossibleCombinaisons([A, C, B, C]);
    result.should.eql(expected);
  });
});

describe('#reduceDatasetDimension', function() {
  describe('on highly redondant problem', function(){
    var problem = [
      [[ 0,  0,  0], 0],
      [[ 0,  0,  1], 1],
      [[ 0,  0,  0], 1],
      [[ 0,  0,  1], 0],
      [[ 1,  1,  0], 0],
      [[ 1,  1,  1], 1],
      [[ 1,  1,  0], 1],
      [[ 1,  1,  1], 0]
    ];
    
    it('should retain 100 percent of the variance', function () {
      var result = nodesvm.reduceDatasetDimension(problem);
      result.retainedVariance.should.be.approximately(1, 1e-5);
      console.log(result.dataset);
    });

    it('should reduce inputs to have a dimension of 2', function () {
      var result = nodesvm.reduceDatasetDimension(problem);
      var reducedInputs = _.map(result.dataset, function(ex) { return ex[0];});
      numeric.dim(reducedInputs).should.eql([8,2]);
      result.newDimension.should.equal(2);
    });

  });

  describe('on a non redondant problem', function(){
    var problem = [
      [[ 0,  0,  0], 0],
      [[ 0,  0,  1], 1],
      [[ 0,  1,  0], 1],
      [[ 0,  1,  1], 0],
      [[ 1,  0,  0], 0],
      [[ 1,  0,  1], 1],
      [[ 1,  1,  0], 1],
      [[ 1,  1,  1], 0]
    ];

    it('should NOT have been reduced if expect 99% of the variance to be retained', function () {
      var result = nodesvm.reduceDatasetDimension(problem, 0.99);
      var reducedInputs = _.map(result.dataset, function(ex) { return ex[0];});
      numeric.dim(reducedInputs).should.eql([8,3]);
      result.newDimension.should.equal(3);
    });

    it('should  have been reduced if expect only 80% of the variance to be retained', function () {
      var result = nodesvm.reduceDatasetDimension(problem, 0.8);
      var reducedInputs = _.map(result.dataset, function(ex) { return ex[0];});
      numeric.dim(reducedInputs).should.eql([8,2]);
      result.newDimension.should.equal(2);
      result.retainedVariance.should.be.approximately(0.83, 1e-2);
    });
  });
});

describe('#reduceInputDimension', function() {
    
  it('should be able to reduce inputs dimensions', function () {
    var input = [1, 0, 0];
    var transformationMatrix = [ 
      [ 1, 1 ],
      [ 1, 1 ],
      [ 1, 1 ] 
    ];
    var expectedOutput = [1, 1];
    
    var output = nodesvm.reduceInputDimension(input, transformationMatrix);
    output.should.eql(expectedOutput);
  });

});