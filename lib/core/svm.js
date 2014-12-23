'use strict';

var assert = require('assert');
var numeric = require('numeric');
var Q = require('q');
var _o = require('mout/object');
var _l = require('mout/lang');
var _a = require('mout/array');


var defaultConfig = require('./config');
var BaseSVM = require('./base-svm');
var gridSearch = require('./grid-search');
var svmTypes = require('./svm-types');
var classification = require('../evaluators/classification');
var regression = require('../evaluators/regression');
var createError = require('../util/create-error');
var normalizeDataset = require('../util/normalize-dataset');
var normalizeInput = require('../util/normalize-input');
var reduce = require('../util/reduce-dataset');

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

    if (!this._config.normalize){
        this._config.mu = _a.take(dims[2], function(){ return 0; });
        this._config.sigma = _a.take(dims[2], function(){ return 1; });
    }
    else {
        var norm = normalizeDataset(dataset);
        this._config.mu = norm.mu;
        this._config.sigma = norm.sigma;
        dataset = norm.dataset;
    }

    if (!this._config.reduce){
        this._config.u = numeric.identity(dims[2]);
        this._retainedVariance = 1;
    }
    else {
        var red = reduce(dataset, this._config.retainedVariance);
        this._config.u = red.U;
        this._retainedVariance = red.retainedVariance;
        dataset = red.dataset;
    }

    // eval all possible combinations using n-fold cross-validation
    return gridSearch(dataset, this._config)
        .then(function(results){
            var best;
            switch (self._config.svmType){
                case svmTypes.C_SVC:
                case svmTypes.NU_SVC:
                case svmTypes.ONE_CLASS:
                    best = _a.max(results, function(r){ return r.report.fscore; });
                    break;
                case svmTypes.EPSILON_SVR:
                case svmTypes.NU_SVR:
                    best = _a.min(results, function(r){ return r.report.mse; });
                    break;
                default :
                    throw createError('not supported classifier', 'ENOTSUPCLF');
            }
            self._baseSvm = new BaseSVM();

            // train a new classifier using entier dataset and best config
            return self._baseSvm.train(dataset, best.config)
                .then(function (model) {
                    model.params = _o.merge(self._config, model.params);
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

    var self = this;
    var predictions = _a.map(testset, function (ex) { return [self.predictSync(ex[0]), ex[1]]; });

    switch (this._config.svmType){
        case svmTypes.C_SVC:
        case svmTypes.NU_SVC:
        case svmTypes.ONE_CLASS:
            return classification.compute(predictions);
        case svmTypes.EPSILON_SVR:
        case svmTypes.NU_SVR:
            return regression.compute(predictions);
        default :
            throw createError('not supported type', 'ENOTSUP');
    }
};

SVM.prototype.getKernelType = function() {
    return this._config.kernelType;
};

SVM.prototype.getSvmType = function() {
    return this._config.svmType;
};

SVM.prototype.normalize = function() {
    return this._config.normalize;
};

SVM.prototype.reduce = function() {
    return this._config.reduce;
};

SVM.prototype.isTrained = function() {
    return !!this._baseSvm ? this._baseSvm.isTrained(): false;
};

SVM.prototype.isTraining = function() {
    return this._training;
};

SVM.prototype.predict = function(inputs){
    assert(this.isTrained());
    var xNorm = normalizeInput(inputs, this._config.mu, this._config.sigma);
    var xRed = numeric.dot(xNorm, this._config.u);
    return this._baseSvm.predict(xRed);
};
SVM.prototype.predictSync = function(inputs){
    assert(this.isTrained());
    var xNorm = normalizeInput(inputs, this._config.mu, this._config.sigma);
    var xRed = numeric.dot(xNorm, this._config.u);
    return this._baseSvm.predictSync(xRed);
};

module.exports = SVM;
