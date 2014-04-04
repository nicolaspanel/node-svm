'use strict';

var should = require ('should'),
    _ = require ('underscore'),
    libsvm = require('../lib/nodesvm');

describe ('Classification Evaluator', function(){
  var evaluator = null;
  var testSet = [
    [[0, 0, 0], 'A'], 
    [[0, 0, 1], 'B'], 
    [[0, 1, 0], 'C'], 
    [[0, 1, 1], 'D'],
    [[1, 0, 0], 'A'], 
    [[1, 0, 1], 'B'], 
    [[1, 1, 0], 'C'], 
    [[1, 1, 1], 'D']
  ];
  var badClassifier = {
    predict: function(state){
      return 'A';
    }, 
    train: function(trainningSet){
      // do nothing
    }
  };
  var perfectClassifier = {
    previsionTable: [['A', 'B'], ['C', 'D']],
    predict: function(state){
      var x1 = state[1];
      var x2 = state[2];
      return this.previsionTable[x1][x2]; // note : independant from state[0]
    }, 
    train: function(trainningSet){
      // do nothing
    }
  };

  
      
  describe ('when evaluates naive classifier', function () {
    
    beforeEach(function () {
      evaluator = new libsvm.ClassificationEvaluator(badClassifier);
    });

    it ('should use only one subset (ie k = 1)', function  (done) {
      evaluator.evaluate(testSet, function(report){
        report.nfold.should.equal(1);
        done();
      }); 
    });
        
    it ('should report an accuracy of 0.25', function(done){
      evaluator.evaluate(testSet, function(report){
        report.accuracy.should.equal(0.25);
        done();
      });
    });

    it ('should report a fscore of 0', function(done){
      evaluator.evaluate(testSet, function(report){
        report.fscore.should.equal(0);
        done();
      });
    });
    
    it ('should report a precision of 0', function(done){
      evaluator.evaluate(testSet, function(report){
        report.precision.should.equal(0);
        done();
      });
    });
    
    it ('should report a recall of 0', function(done){
      evaluator.evaluate(testSet, function(report){
        report.recall.should.equal(0);
        done();
      });
    });
    
    it ('should report a recall of 1 for class \'A\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['A'].recall.should.equal(1);
        done();
      });
    });
    
    it ('should report a precision of 0.25 for class \'A\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['A'].precision.should.equal(0.25);
        done();
      });
    });
    
    it ('should report a fscore of 0.4 for class \'A\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['A'].fscore.should.equal(0.4);
        done();
      });
    });

    it ('should report a recall of 0 for class \'B\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['B'].recall.should.equal(0);
        done();
      });
    });
    
    it ('should report a precision of 0 for class \'B\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['B'].precision.should.equal(0);
        done();
      });
    });
    
    it ('should report a fscore of 0 for class \'B\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['B'].fscore.should.equal(0);
        done();
      });
    });

  });

  describe ('when perform n-fold cross validation on perfect classifier', function(){
    beforeEach(function () {
      evaluator = new libsvm.ClassificationEvaluator(perfectClassifier);
    });

    it ('should report an accuracy of 1', function(done){
      evaluator.performNFoldCrossValidation(8, testSet, function(report){
        report.accuracy.should.equal(1);
        report.fscore.should.equal(1);
        report.precision.should.equal(1);
        report.recall.should.equal(1);
        done();
      });
    });

    it ('should report a recall of 1 for all classes', function(done){
      evaluator.performNFoldCrossValidation(1, testSet, function(report){
        report.subsetsReports.forEach(function(subset){
          ['A', 'B', 'C', 'D'].forEach(function(label){
            subset.classReports[label].recall.should.equal(1);
          }); 
        });
        done();
      });
        
    });
      
    
    it ('should report a precision of 1 for all classes', function(done){
      evaluator.performNFoldCrossValidation(1, testSet, function(report){
        report.subsetsReports.forEach(function(subset){
          ['A', 'B', 'C', 'D'].forEach(function(label){
            subset.classReports[label].precision.should.equal(1);
          }); 
        });
        done();
      });
    });
      
    it ('should report a fscore of 1 for all classes', function(done){
      evaluator.performNFoldCrossValidation(1, testSet, function(report){
        report.subsetsReports.forEach(function(subset){
          ['A', 'B', 'C', 'D'].forEach(function(label){
            subset.classReports[label].fscore.should.equal(1);
          }); 
        });
        done();
      });
    });  
  });
});
