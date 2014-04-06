'use strict';

var should = require ('should'),
    _ = require ('underscore'),
    libsvm = require('../lib/nodesvm');

var evaluator = null;
var testSet = [
  [[0, 0, 0], '0'], 
  [[0, 0, 1], '1'], 
  [[0, 1, 0], '2'], 
  [[0, 1, 1], '3'],
  [[1, 0, 0], '0'], 
  [[1, 0, 1], '1'], 
  [[1, 1, 0], '2'], 
  [[1, 1, 1], '3']
];
var badClassifier = {
  predict: function(state){
    return '0';
  }, 
  train: function(trainingSet){
    // do nothing
  },
  trainAsync: function(trainingSet, cb){
    cb();// do nothing
  }
};
var perfectClassifier = {
  previsionTable: [['0', '1'], ['2', '3']],
  predict: function(state){
    var x1 = state[1];
    var x2 = state[2];
    return this.previsionTable[x1][x2]; // note : independant from state[0]
  }, 
  train: function(trainingSet){
    // do nothing
  },
  trainAsync: function(trainingSet, cb){
    cb();// do nothing
  }
};

describe ('Classification Evaluator', function(){
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
    
    it ('should report a recall of 1 for class \'0\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['0'].recall.should.equal(1);
        done();
      });
    });
    
    it ('should report a precision of 0.25 for class \'0\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['0'].precision.should.equal(0.25);
        done();
      });
    });
    
    it ('should report a fscore of 0.4 for class \'0\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['0'].fscore.should.equal(0.4);
        done();
      });
    });

    it ('should report a recall of 0 for class \'1\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['1'].recall.should.equal(0);
        done();
      });
    });
    
    it ('should report a precision of 0 for class \'1\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['1'].precision.should.equal(0);
        done();
      });
    });
    
    it ('should report a fscore of 0 for class \'1\'', function(done){
      evaluator.evaluate(testSet, function(report){
        report.subsetsReports[0].classReports['1'].fscore.should.equal(0);
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
          ['0', '1', '2', '3'].forEach(function(label){
            subset.classReports[label].recall.should.equal(1);
          }); 
        });
        done();
      });
        
    });
      
    
    it ('should report a precision of 1 for all classes', function(done){
      evaluator.performNFoldCrossValidation(1, testSet, function(report){
        report.subsetsReports.forEach(function(subset){
          ['0', '1', '2', '3'].forEach(function(label){
            subset.classReports[label].precision.should.equal(1);
          }); 
        });
        done();
      });
    });
      
    it ('should report a fscore of 1 for all classes', function(done){
      evaluator.performNFoldCrossValidation(1, testSet, function(report){
        report.subsetsReports.forEach(function(subset){
          ['0', '1', '2', '3'].forEach(function(label){
            subset.classReports[label].fscore.should.equal(1);
          }); 
        });
        done();
      });
    });  
  });
});

describe ('Regression Evaluator', function(){
  describe ('when evaluates naive classifier', function () {

    beforeEach(function () {
      evaluator = new libsvm.RegressionEvaluator(badClassifier);
    });

    it ('should use only one subset (ie k = 1)', function  (done) {
      evaluator.evaluate(testSet, function(report){
        report.nfold.should.equal(1);
        done();
      }); 
    });
        
    it ('should report a mse of 3.5', function(done){
      evaluator.evaluate(testSet, function(report){
        report.mse.should.equal(3.5);
        done();
      });
    });

  });

  describe ('when perform n-fold cross validation on perfect classifier', function(){
    beforeEach(function () {
      evaluator = new libsvm.RegressionEvaluator(perfectClassifier);
    });

    it ('should report a mse of 0', function(done){
      evaluator.performNFoldCrossValidation(4, testSet, function(report){
        report.mse.should.equal(0);
        done();
      });
    }); 
  });
});

