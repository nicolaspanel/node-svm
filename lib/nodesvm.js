'use strict';

var svm = require('./svm'),
    helpers = require('./helpers'),
    classificator = require('./evaluators');


exports.SvmTypes = svm.SvmTypes;
exports.KernelTypes = svm.KernelTypes;
exports.LinearKernel = svm.LinearKernel;
exports.PolynomialKernel = svm.PolynomialKernel;
exports.RadialBasisFunctionKernel = svm.RadialBasisFunctionKernel;
exports.SigmoidKernel = svm.SigmoidKernel;
exports.SVM = svm.SVM;

exports.readProblemAsync = helpers.readProblemAsync;
exports.meanNormalize = helpers.meanNormalize;
exports.readAndNormalizeProblemAsync = helpers.readAndNormalizeProblemAsync;
exports.performNFoldCrossValidation = helpers.performNFoldCrossValidation;
exports.evaluateSvm = helpers.evaluateSvm;

exports.ClassificationEvaluator = classificator.ClassificationEvaluator;