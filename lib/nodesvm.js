'use strict';

var svm = require('./svm'),
    helpers = require('./helpers'),
    classificator = require('./evaluators');

// svm.js
exports.SvmTypes = svm.SvmTypes;
exports.KernelTypes = svm.KernelTypes;
exports.LinearKernel = svm.LinearKernel;
exports.PolynomialKernel = svm.PolynomialKernel;
exports.RadialBasisFunctionKernel = svm.RadialBasisFunctionKernel;
exports.SigmoidKernel = svm.SigmoidKernel;
exports.SVM = svm.SVM;

// helpers.js
exports.readDatasetAsync = helpers.readDatasetAsync;
exports.meanNormalizeDataSet = helpers.meanNormalizeDataSet;
exports.readAndNormalizeDatasetAsync = helpers.readAndNormalizeDatasetAsync;
exports.performNFoldCrossValidation = helpers.performNFoldCrossValidation;
exports.evaluateSvm = helpers.evaluateSvm;
exports.meanNormalizeInput = helpers.meanNormalizeInput;
exports.findBestParameters = helpers.findBestParameters;
exports.findAllPossibleCombinaisons = helpers.findAllPossibleCombinaisons;
exports.reduceInputDimension = helpers.reduceInputDimension;
exports.reduceDatasetDimension = helpers.reduceDatasetDimension;

// evaluators.js
exports.ClassificationEvaluator = classificator.ClassificationEvaluator;
exports.RegressionEvaluator = classificator.RegressionEvaluator;