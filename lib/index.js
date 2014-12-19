'use strict';

var util = require('util'),
    svmTypes = require('./core/svm-types'),
    kernelTypes = require('./core/kernel-types'),
    SVM = require('./core/svm'),
    helpers = require('./helpers');


function CSVC(config, model) {
    config.svmType = svmTypes.C_SVC;
    SVM.call(this, config, model);
}
util.inherits(CSVC, SVM);

function NuSVC(config, model) {
    config.svmType = svmTypes.NU_SVC;
    SVM.call(this, config, model);
}
util.inherits(NuSVC, SVM);

function EpsilonSVR(config, model) {
    config.svmType = svmTypes.EPSILON_SVR;
    SVM.call(this, config, model);
}
util.inherits(EpsilonSVR, SVM);

function NuSVR(config, model) {
    config.svmType = svmTypes.NU_SVR;
    SVM.call(this, config, model);
}
util.inherits(NuSVR, SVM);

function restore(model) {
    return new SVM({}, model);
}

module.exports = {
    svmTypes: svmTypes,
    kernelTypes: kernelTypes,

    // helpers
    read: helpers.read,
    restore: restore,

    SVM: SVM,
    CSVC: CSVC,
    NuSVC: NuSVC,
    EpsilonSVR: EpsilonSVR,
    NuSVR: NuSVR
};


