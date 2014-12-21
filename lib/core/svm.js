'use strict';

var assert = require('assert');
var numeric = require('numeric');
var Q = require('q');
var _o = require('mout/object');
var _l = require('mout/lang');
var _a = require('mout/array');
var defaultConfig = require('./config');
var utils = require('../util');
var BaseSVM = require('./base-svm');
var gridSearch = require('./grid-search');
var svmTypes = require('./svm-types');
var classification = require('../evaluators/classification');
var regression = require('../evaluators/regression');

var SVM = function(config, model){
    this._config = _l.deepClone(defaultConfig(config));
    if (model){
        this._restore(model);
    }
};

SVM.prototype._restore = function (model) {
    var self = this;
    this._baseSvm = BaseSVM.restore(model);
    _o.forOwn(model.params, function(val, key){
        self._config[key] = val;
    });
};

SVM.prototype.train = function(dataset) {
    var self= this;
    this._training = true;
    var dims = numeric.dim(dataset);
    assert(dims[0]>0 && dims[1] === 2 && dims[2]>0 , 'dataset must be an list of [X,y] tuples');

    var nDataset =  this._normaliseDataset(dataset, dims);
    var rDataset = this._reduceDataset(nDataset, dims);

    // eval all possible combinations using n-fold cross-validation
    return gridSearch(rDataset, this._config)
        .then(function(results){
            var best;
            switch (self._config.svmType){
                case svmTypes.C_SVC:
                case svmTypes.NU_SVC:
                    best = _a.max(results, function(r){ return r.report.fscore; });
                    break;
                case svmTypes.EPSILON_SVR:
                case svmTypes.NU_SVR:
                    best = _a.min(results, function(r){ return r.report.mse; });
                    break;
                default :
                    throw utils.createError('not supported classifier', 'ENOTSUPCLF');
            }
            self._baseSvm = new BaseSVM();

            // train a new classifier using entier dataset and best config
            return self._baseSvm.train(dataset, best.config)
                .then(function (model) {
                    _o.mixIn(model.params, {
                        mu: self._mu,
                        sigma : self._sigma,
                        u : self._u
                    });
                    _o.mixIn(best.report, {
                        retainedVariance : self._retainedVariance
                    });
                    return [model, best.report];
                });
        })
        .fin(function () {
            self._training = false;
        });
};

SVM.prototype.evaluate = function (testset) {
    assert(this.isTrained(), 'train classifier first');
    var dims = numeric.dim(testset);
    assert(dims[0]>0 && dims[1] === 2 && dims[2]>0 , 'testset must be an list of [X,y] tuples');

    var nTestset =  this._normaliseDataset(testset, dims);
    var rTestset = this._reduceDataset(nTestset, dims);
    switch (this._config.svmType){
        case svmTypes.C_SVC:
        case svmTypes.NU_SVC:
            return classification.evaluate(rTestset, this._baseSvm);
        case svmTypes.EPSILON_SVR:
        case svmTypes.NU_SVR:
            return regression.evaluate(rTestset, this._baseSvm);
        default :
            throw utils.createError('not supported type', 'ENOTSUP');
    }
};

SVM.prototype._normaliseDataset = function(dataset, dims){

    if (!this.normalize()){
        this._mu = numeric.rep([dims[2]], 0);
        this._sigma = numeric.rep([dims[2]], 1);
        return dataset;
    }
    else {
        var res = utils.normalizeDataset(dataset);
        this._mu = res.mu;
        this._sigma = res.sigma;
        return res.dataset;
    }
};

SVM.prototype._reduceDataset = function(dataset, dims){

    if (!this.reduce()){
        this._u = numeric.diag(dims[2]);
        this._retainedVariance = 1;
        return dataset;
    }
    else {
        var res = utils.reduce(dataset, this._config.retainedVariance);
        this._u = res.U;
        this._retainedVariance = res.retainedVariance;
        return res.dataset;
    }
};

SVM.prototype.getKernelType = function() {
    return this._config.kernelType;
};

SVM.prototype.getSvmType = function() {
    return this._config.svmType;
};

SVM.prototype.normalize = function() {
    return _o.has(this, '_normalize') ? this._normalize : this._config.normalize;
};

SVM.prototype.reduce = function() {
    return _o.has(this, '_reduce') ? this._reduce : this._config.reduce;
};

SVM.prototype.isTrained = function() {
    return !!this._baseSvm ? this._baseSvm.isTrained(): false;
};

SVM.prototype.isTraining = function() {
    return this._training;
};

SVM.prototype.predict = function(inputs){
    return this._baseSvm.predict(inputs);
};
SVM.prototype.predictSync = function(inputs){
    return this._baseSvm.predictSync(inputs);
};

module.exports = SVM;
