'use strict';

var _a = require('mout/array');
var Q = require('q');
var assert = require('assert');
var numeric= require('numeric');

/**
 NOTICE : this function assumes your predictor is already trained
 */
//RegressionEvaluator.prototype.evaluate = function(testSet, evalFn){
//    if (typeof callback === 'function'){
//        this.once('done', callback);
//    }
//    this.performNFoldCrossValidation(1, testSet);
//};
//
//RegressionEvaluator.prototype.performNFoldCrossValidation = function(nfold, dataSet, callback) {
//    if (typeof callback === 'function'){
//        this.once('done', callback);
//    }
//    var data = _.shuffle(dataSet);
//
//    var nbExamplePerSubset = Math.floor(data.length / nfold);
//    var dataSubsets = [];
//    var k = 0;
//    var i = 0, j=0;
//    for (i = 0; i < nfold; i++){
//        var subset = data.slice(k, k + nbExamplePerSubset);
//        k += nbExamplePerSubset;
//        dataSubsets.push(subset);
//    }
//
//    var _sets = [];
//    for (i = 0; i < nfold; i++){
//        var iTrainningSet = [];
//
//        for (j = 0; j < nfold; j++){
//            if(j !== i){
//                iTrainningSet = iTrainningSet.concat(dataSubsets[j]);
//            }
//        }
//
//        _sets.push({
//            trainning: iTrainningSet,
//            test: dataSubsets[i]
//        });
//    }
//
//    var setReports = [];
//    var self = this;
//    async.each(_sets, function(set, done){
//        self._trainAndEvaluateSet(set, function (r) {
//            setReports.push(r);
//            done();
//        });
//    }, function(err){
//        var mses = _.map(setReports, function (r) { return r.mse; });
//        var sumMSE = _.reduce(mses, function (memo, val) { return memo + val; });
//        self.emit('done', {
//            nfold: nfold,
//            mse: sumMSE / nfold,
//            subsetsReports: setReports
//        });
//    });
//};

function compute(predictions){
    var n = predictions.length,
        errors =  _a.map(predictions, function(p){ return p[0] - p[1]; }),
        avgError = _a.reduce(errors, function(sum, e){ return sum + e; }, 0) / n,
        variance = _a.reduce(errors, function(sum, e){ return sum + Math.pow(e - avgError, 2); }, 0) / n;

    return {
        //predictions: predictions,
        mse: _a.reduce(errors, function(sum, e){ return sum + Math.pow(e, 2); }, 0) / n,
        std: Math.pow(variance, 0.5),
        mean: avgError
    };
}

function evaluate(testSet, clf) {
    var dims = numeric.dim(testSet);
    assert(dims[0]>0 && dims[1] === 2 && dims[2]>0 , 'test set must be an list of [X,y] tuples');

    var predictions = testSet.map(function(test){
        return [clf.predictSync(test[0]), test[1]];
    });
    return compute(predictions);

}

module.exports = {
    evaluate: evaluate,
    compute: compute
};