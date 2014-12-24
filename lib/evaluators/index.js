'use strict';

var svmTypes = require('../core/svm-types');
var createError = require('../util/create-error');

var classification = require('./classification');
var regression = require('./regression');

module.exports = {
    classification: classification,
    regression: regression,
    getDefault: function(config){
        switch (config.svmType){
            case svmTypes.C_SVC:
            case svmTypes.NU_SVC:
            case svmTypes.ONE_CLASS:
                return classification;
            case svmTypes.EPSILON_SVR:
            case svmTypes.NU_SVR:
                return regression;
            default :
                throw createError('No evaluator found for given configuration', 'ENOTFOUND');
        }
    }
};