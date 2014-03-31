'use strict';

var assert = require('assert'), 
    should = require('should'),
    _ = require('underscore'),
    libsvm = require('../lib/nodesvm');

var xorProblem = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 }
];
var xorNormProblem = [
  { x: [-1, -1], y: 0 },
  { x: [-1,  1], y: 1 },
  { x: [ 1, -1], y: 1 },
  { x: [ 1,  1], y: 0 }
];
describe('libsvm', function(){
  
  describe('Linear kernel', function(){
    var kernel = null;
    beforeEach(function(){
      kernel = new libsvm.LinearKernel();
    });
    it('should have type set to 0', function(){
      kernel.kernelType.should.equal(0);
    });
  });

  describe('Polynomial kernel', function(){
    var kernel = null;
    beforeEach(function(){
      kernel = new libsvm.PolynomialKernel(3, 4, 5);
    });
    it('should have type set to 1', function(){
      kernel.kernelType.should.equal(1);
    });
    it('should have degree set to 3', function(){
      kernel.degree.should.equal(3);
    });
    it('should have gamma set to 4', function(){
      kernel.gamma.should.equal(4);
    });
    it('should have r set to 5', function(){
      kernel.r.should.equal(5);
    });
  });

  describe('RBF kernel', function(){
    var kernel = null;
    beforeEach(function(){
      kernel = new libsvm.RadialBasisFunctionKernel(3);
    });
    it('should have type set to 2', function(){
      kernel.kernelType.should.equal(2);
    });
    it('should have gamma set to 3', function(){
      kernel.gamma.should.equal(3);
    });
  });

  describe('Sigmoid kernel', function(){
    var kernel = null;
    beforeEach(function(){
      kernel = new libsvm.SigmoidKernel(3, 4);
    });
    it('should have type set to 3', function(){
      kernel.kernelType.should.equal(3);
    });
    it('should have gamma set to 3', function(){
      kernel.gamma.should.equal(3);
    });
    it('should have r set to 4', function(){
      kernel.r.should.equal(4);
    });
  });

  describe('using SVM with bad parameters', function(){
    it('should throw an error during initialization', function(){
      var testFunc = function(){
        var svm = new libsvm.SVM({
          type: libsvm.SvmTypes.C_SVC,
          kernel: new libsvm.RadialBasisFunctionKernel(2),
          C: 0 // c must be > 0
        });
      };
      testFunc.should.throw();
    });
  });

  describe('using NU_SVC with Sigmoid Kernel', function(){
    var svm = null;
    beforeEach(function(){
      svm = new libsvm.SVM({
        type: libsvm.SvmTypes.NU_SVC,
        kernel: new libsvm.SigmoidKernel(2),
        nu: 0.4
      });
    });
    
    it('should have a reference to the NodeSVM obj', function(){
      svm._nodeSvm.should.be.ok;
    });

    it('should use Sigmoid kernel ', function(){
      svm.getKernelType().should.eql('SIGMOID');
    });

    it('should use NU_SVC classificator ', function(){
      svm.getSvmType().should.eql('NU_SVC');
    });
  });

  describe('using C_SVC with RBF Kernel', function(){
    var svm = null;
    beforeEach(function(){
      svm = new libsvm.SVM({
        type: libsvm.SvmTypes.C_SVC,
        kernel: new libsvm.RadialBasisFunctionKernel(2),
        C: 2
      });
    });

    it('should train XOR with no error', function(){
      var testFunc = function(){
        svm.train(xorProblem);
      };
      testFunc.should.not.throw();
      
    });
    
    describe('once trained with xor dataset', function(){
      beforeEach(function(){
        svm.train(xorProblem);
      });
      
      it('should be able to return class labels', function(){
        svm.labels.should.eql([0, 1]);
      });

      it('should be able to predict classes', function(){
        xorProblem.forEach(function(ex){
          [0,1].should.containEql(svm.predict(ex.x));  // ie mean y E {-1;1}
        });
      });


      it('should be able to predict Async', function(done){
        svm.predictAsync([0, 0], function(value){
          [0,1].should.containEql(value);
          done();
        });
      });

      it('should be able to predict probabilities', function(){
        var probs = svm.predictProbabilities([0, 0]);
        (probs[0] + probs[1]).should.be.approximately(1, 1e-5);
      });

      // it('should evaluate an accuracy of 100%', function(done){
      //   svm.getAccuracy(xorProblem, function(accuracy){
      //     accuracy.should.be.approximately(1, 1e-5);
      //     done();
      //   });
      // });
    });
  });

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

  describe('#meanNormalize', function(){  
    
    it('should be able to Mean Normalize the xor problem', function () {
      var result = libsvm.meanNormalize({problem: xorProblem});
      result.mu.should.eql([0.5, 0.5]);
      result.sigma.should.eql([0.5,0.5]);
      result.problem.should.eql(xorNormProblem);
    });
    
    it('should be able to Mean Normalize the xor problem with custom mu and sigma', function () {
      var result = libsvm.meanNormalize({problem: xorProblem, mu: [0, 0], sigma: [1, 1]});
      result.problem.should.eql(xorProblem); // no changes
    });
  });

  describe('#readAndNormalizeProblemAsync', function(){  
    it('should be able to read and mean normalize the xor problem', function (done) {
      libsvm.readAndNormalizeProblemAsync('./examples/datasets/xor.ds', function(result){
        result.mu.should.eql([0.5, 0.5]);
        result.sigma.should.eql([0.5,0.5]);
        result.problem.should.eql(xorNormProblem);
        done();
      });      
    });
  });
});
