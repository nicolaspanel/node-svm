'use strict';

var assert = require('assert'), 
    should = require('should'),
    _ = require('underscore'),
    lib = require('../lib/node-svm');
    
describe('SVM', function(){
  var svm = null;
  var xorProblem = [
    { x: [-1, -1], y: -1 },
    { x: [-1, 1], y: 1 },
    { x: [1, -1], y: 1 },
    { x: [1, 1], y: -1 }
  ];
  
  describe('when parameters are not properly defined', function(){
    beforeEach(function(){
      svm = new lib.SVM();
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
      svm = new lib.SVM({
          type: lib.SvmTypes.C_SVC,
          kernel: new lib.RadialBasisFunctionKernel(1),
          C: 0.8
      });
    });
    
    it('should have a reference to the NodeSVM obj ', function(){
      svm._node_svm.should.be.ok;
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
        svm.labels.should.eql([1, -1]);
      });

      it('should be able to predict classes', function(){
        xorProblem.forEach(function(ex){
          [-1,1].should.containEql(svm.predict(ex.x));  // ie mean y E {-1;1}
        });
      });


      it('should be able to predict Async', function(done){
        svm.predictAsync([-1, -1], function(value){
          [-1,1].should.containEql(value);
          done();
        });
      });

      it('should be able to predict probabilities', function(){
        var probs = svm.predictProbabilities([-1, -1]);
        (probs[-1] + probs[1]).should.be.approximately(1, 1e-5);
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
