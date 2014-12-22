'use strict';

var _a = require('mout/array');
var Q = require('q');
var assert = require('assert');
var numeric= require('numeric');

function compute(predictions){
    var n = predictions.length,
        errors =  _a.map(predictions, function(p){ return p[0] - p[1]; }),
        avgError = _a.reduce(errors, function(sum, e){ return sum + e; }, 0) / n,
        variance = _a.reduce(errors, function(sum, e){ return sum + Math.pow(e - avgError, 2); }, 0) / n;

    return {
        mse: _a.reduce(errors, function(sum, e){ return sum + Math.pow(e, 2); }, 0) / n,
        std: Math.pow(variance, 0.5),
        mean: avgError,
        size: predictions.length
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
