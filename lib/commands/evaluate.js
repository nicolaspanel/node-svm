'use strict';

var Q = require('q');
var path = require('path');
var fs = require('graceful-fs');
var _a = require('mout/array');

var svmTypes = require('../core/svm-types');
var SVM = require('../core/svm');
var createError = require('../util/create-error');
var cli = require('../util/cli');
var read = require('../util/read-dataset');


function evaluate(logger, options) {

    if (!options.pathToModel){
        return Q.reject(createError('<model file> required', 'EINVARGUMENTS', {
            command: 'evaluate'
        }));
    }
    if (!options.pathToTestset){
        return Q.reject(createError('<test set file> required', 'EINVARGUMENTS', {
            command: 'evaluate'
        }));
    }
    var modelPath = path.resolve(options.cwd || '.', options.pathToModel);
    var testsetPath = path.resolve(options.cwd || '.', options.pathToTestset);
    delete options.pathToModel;
    delete options.pathToTestset;

    return Q.all([
        readModel(modelPath),
        readTestset(testsetPath)
    ]).spread(function (model, testset) {
        var svm = new SVM({}, model);
        var report = svm.evaluate(testset);

        var template;
        switch (svm.getSvmType()){
            case svmTypes.C_SVC:
            case svmTypes.NU_SVC:
            case svmTypes.ONE_CLASS:
                template = 'classification-report';
                break;
            case svmTypes.EPSILON_SVR:
            case svmTypes.NU_SVR:
                template = 'regression-report';
                break;
        }

        logger.info('template', 'report', {
            template: template,
            json: report
        });
        return report;
    });
}


// ----------- helpers
function readModel(file) {
   return checkFileExists(file)
        .then(function (exists) {
            if (!exists) {
                throw createError('Model not found', 'ENOTFOUND');
            }
        })
       .then(function () {
           return Q.nfcall(fs.readFile, file)
               .then(function (data) {
                   return JSON.parse(data.toString());
               });
       });
}

function readTestset(file) {
    return checkFileExists(file)
        .then(function (exists) {
            if (!exists) {
                throw createError('Test set not found', 'ENOTFOUND');
            }
        })
        .then(function () {
            return read(file);
        });
}

function checkFileExists(file){
    return Q.promise(function (resolve) {
        fs.exists(file, resolve);
    });
}

// -------------------

evaluate.line = function (logger, argv) {
    var options = cli.readOptions(argv);
    if (options.argv.remain[1]){
        options.pathToModel = options.argv.remain[1];
        if (options.argv.remain[2]){
            options.pathToTestset = options.argv.remain[2];
        }
    }
    delete options.argv;
    return evaluate(logger , options);
};

evaluate.completion = function () {
    // TODO
};

module.exports = evaluate;