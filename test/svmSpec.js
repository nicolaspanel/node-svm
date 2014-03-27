'use strict';

var assert = require('assert'), 
    should = require('should'),
    lib = require('../lib/node-svm');
    
describe('SVM', function(){
  var svm = null;
  var xorProblem = [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 }
  ];

  beforeEach(function(){
    svm = new lib.SVM({
        type: lib.SvmTypes.C_SVC,
        kernel: new lib.LinearKernel()
    });
  });
  
  it('should have  type set to C_SVC ', function(){
    svm.svmType.should.eql('C_SVC');
  });


  it('should use linear kernel by default ', function(){
    svm.kernel.type.should.eql('LINEAR');
  });
  
  it('can be trained', function(done){
    svm.train(xorProblem, function(){
        done();
    });
  });

    
});