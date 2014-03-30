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
describe('SVM', function(){
  var svm = null;
  
  describe('when parameters are not properly defined', function(){
    beforeEach(function(){
      svm = new libsvm.SVM();
    });
    it('should report an error during training', function(done){
      svm.train(xorProblem, function (err) {
        console.log(err);
        err.should.be.ok;
        done();
      });
    });
  });

  describe('using C_SVC with RBF Kernel', function(){
    
    beforeEach(function(){
      svm = new libsvm.SVM({
        type: libsvm.SvmTypes.C_SVC,
        kernel: new libsvm.RadialBasisFunctionKernel(1),
        C: 0.8
      });
    });
    
    it('should have a reference to the NodeSVM obj ', function(){
      svm._nodeSvm.should.be.ok;
    });

    it('should have type set to C_SVC ', function(){
      svm.svmType.should.eql('C_SVC');
    });

    it('should use RBF kernel ', function(){
      svm.kernel.type.should.eql('RBF');
    });

    it('should train XOR with no error', function(done){
      svm.train(xorProblem, function (err) {
        done();
      });
    });
    
    describe('once trained', function(){
      beforeEach(function(done){
        svm.train(xorProblem, function () {
          done();
        });
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

      it('should be able to predict accuracy on a test problem', function(done){
        svm.getAccuracy(xorProblem, function(accuracy){
          accuracy.should.be.within(0, 1);
          done();
        });
      });
    });
  });
});

describe('#readProblem', function(){  
  it('should be able to read the xor problem', function (done) {
    libsvm.readProblem('./examples/datasets/xor.ds', function(problem, nbFeature){
      nbFeature.should.equal(2);
      problem.length.should.equal(4);
      problem.should.eql(xorProblem);
      done();
    });
  });
  it('should be able to read the svmguide problem in less than 200ms', function (done) {
    this.timeout(200);
    libsvm.readProblem('./examples/datasets/svmguide1.ds', function(problem, nbFeature){
      nbFeature.should.equal(4);
      problem.length.should.equal(3089);
      done();
    });
  });
});
