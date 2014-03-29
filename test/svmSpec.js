'use strict';

var assert = require('assert'), 
    should = require('should'),
    lib = require('../lib/node-svm');
    
describe('SVM', function(){
  var svm = null;
  var xorProblem = [
    { x: [-1, -1], y: 0 },
    { x: [-1, 1], y: 1 },
    { x: [1, -1], y: 1 },
    { x: [1, 1], y: 0 }
  ];
  
  describe('when parameters are not properly defined', function(){
    beforeEach(function(){
      svm = new lib.SVM();
    });
    it('should report error during training', function(done){
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
        err.should.not.be.ok;
        done();
      });
    });
    describe('once trained', function(){
      beforeEach(function(done){
        svm.train(xorProblem, function () {
          done();
        });
      });

      it('should be able to predict', function(done){
        svm.predict([-1, -1], function (prediciton) {
          prediciton.should.be.within(0, 1);
          done();
        });
      });

    });
  });
});