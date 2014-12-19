'use strict';

var _l = require('mout/lang');
var _o = require('mout/object');
var _a = require('mout/array');
var assert = require('assert');
var Q = require('q');
var numeric= require('numeric');

var svmTypes = require('./svm-types');
var kernelTypes = require('./kernel-types');
var helpers = require('../helpers');
var classification = require('../evaluators/classification');
var regression = require('../evaluators/regression');
var createError = require('../util/create-error');
var BaseSVM = require('./base-svm');
var defaultConfig = require('./config');


module.exports = function(dataset, config){
    var deferred = Q.defer();
    // default options
    var dims = numeric.dim(dataset);

    assert(dims[0]>0 && dims[1] === 2 && dims[2]>0 , 'dataset must be a list of [X,y] tuples');

    var params = _l.deepClone(defaultConfig(config));

    var combs = helpers.crossCombinations([
        params.c,
        params.gamma,
        params.epsilon,
        params.nu,
        params.degree,
        params.r
    ]);

    var subsets = helpers.splitDataSet(dataset, params.kFold);
    var evaluator;
    switch (params.svmType){
        case svmTypes.C_SVC:
        case svmTypes.NU_SVC:
            evaluator = classification;
            break;
        case svmTypes.EPSILON_SVR:
        case svmTypes.NU_SVR:
            evaluator = regression;
            break;
        default :
            throw createError('not supported type', 'ENOTSUP');
    }
    var promises = combs.map(function (comb) {
        var cParams = _o.merge(params, {
            c : comb[0],
            gamma: comb[1],
            epsilon: comb[2],
            nu: comb[3],
            degree: comb[4],
            r: comb[5]
        });
        var cPromises = subsets.map(function(ss){
            var clf = new BaseSVM();
            return clf.train(ss.train, cParams)
                .then(function(model){
                    return _a.map(ss.test, function(test){
                        return [clf.predictSync(test[0]), test[1]];
                    });
                });
        });
        return Q.all(cPromises)
            .then(function (predictions) {
                predictions = _a.flatten(predictions, 1);
                var report = evaluator.compute(predictions);
                delete report.predictions;

                return {
                    config:cParams,
                    report: report
                };
            }).fail(function (err ) {
                throw err;
            });
    });
    Q.all(promises).then(function (results) {
        deferred.resolve(results);
    });

    return deferred.promise;
};