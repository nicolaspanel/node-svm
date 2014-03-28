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
    { x: [1, 1], y: 0 },
    { x: [-1, -1], y: 0 },
    { x: [-1, 1], y: 1 },
    { x: [1, -1], y: 1 },
    { x: [1, 1], y: 0 }
  ];
  describe('using C_SVC with LinearKernel', function(){
    
    beforeEach(function(){
      svm = new lib.SVM({
          type: lib.SvmTypes.C_SVC,
          kernel: new lib.LinearKernel()
      });
    });
      
    it('should have type set to C_SVC ', function(){
      svm.svmType.should.eql('C_SVC');
    });

    it('should use linear kernel by default ', function(){
      svm.kernel.type.should.eql('LINEAR');
    });

    it('can be train without error', function(done){
      svm.train(xorProblem, function(err){
          err.should.eql('');
          done();
      });
    });
  });
});