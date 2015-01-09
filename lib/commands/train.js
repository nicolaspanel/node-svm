'use strict';

var Q = require('q');
var path = require('path');
var chalk = require('chalk');
var fs = require('fs');
var _o = require('mout/object');
var _a = require('mout/array');
var _l = require('mout/lang');
var _s = require('mout/string');

var defaultConfig = require('../core/config');
var svmTypes = require('../core/svm-types');
var kernelTypes = require('../core/kernel-types');
var SVM = require('../core/svm');
var createError = require('../util/create-error');
var cli = require('../util/cli');
var read = require('../util/read-dataset');

var svmTypesString = _o.keys(svmTypes);
var kernelTypesString = _o.keys(kernelTypes);

function train(logger, options) {
    var config;
    if (!options.pathToDataset){
        return Q.reject(createError('<dataset file> required', 'EINVARGUMENTS'));
    }
    var datasetPath = path.resolve(options.cwd || '.', options.pathToDataset);
    delete options.pathToDataset;

    var defaults = defaultConfig();
    var interactive = _o.has(options, 'interactive') ? !!options.interactive:  defaults.interactive;
    return checkFileExists(datasetPath)
        .then(function (exists) {
            if (!exists) {
                throw createError('Data set not found', 'ENOTFOUND');
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'svmType')){
                return askSvmType(logger, defaults.svmType)
                    .then(function (svmType) { options.svmType = svmType; });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'kernelType')){
                return askKernelType(logger, defaults.kernelType)
                    .then(function (kernelType) { options.kernelType = kernelType; });
            }
        })
        .then(function () {
            // once we know which svm/kernel are used
            // we can determine which parameters to ask
            config = defaultConfig(options);
        })
        .then(function () {
            if(interactive && !_o.has(options, 'c') && req('c', options)){
                return askCost(logger, config.c)
                    .then(function (c) { config.c = c; });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'gamma') && req('gamma', options)){
                return askGamma(logger, config.gamma)
                    .then(function (gamma) { config.gamma = gamma; });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'degree') && req('degree', options)){
                return askDegree(logger, config.degree)
                    .then(function (degree) { config.degree = degree; });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'r') && req('r', options)){
                return askR(logger, config.r)
                    .then(function (r) { config.r = r; });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'nu') && req('nu', options)){
                return askNu(logger, config.nu)
                    .then(function (nu) { config.nu = nu; });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'epsilon') && req('epsilon', options)){
                return askEpsilon(logger, config.epsilon)
                    .then(function (epsilon) { config.epsilon = epsilon; });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'kFold')){
                return askKFold(logger, defaults.kFold)
                    .then(function (kFold) {
                        config.kFold = kFold;
                    });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'normalize')){
                return askNormalize(logger, defaults.normalize)
                    .then(function (normalize) { config.normalize = normalize; });
            }
        })
        .then(function () {
            if(interactive && !_o.has(options, 'reduce')){
                return askReduce(logger, defaults.reduce)
                    .then(function (reduce) {
                        config.reduce = reduce;
                        return reduce;
                    });
            }
            else if (_o.has(options, 'reduce')) {
                return options.reduce;
            }
            else {
                return defaults.reduce;
            }
        })
        .then(function (reduce) {
            if(interactive && reduce && !_o.has(options, 'retainedVariance')){
                return askRetainedVariance(logger, defaults.retainedVariance)
                    .then(function (retained) {
                        config.retainedVariance = retained;
                    });
            }
        })
        .then(function () {
            // ask user confirmation if interactions are enabled
            var deferred = Q.defer();
            if (interactive) {
                askConfirmation(logger, config).then(function (good) {
                    deferred.resolve(good);
                });
            }
            else {
                deferred.resolve(true);
            }
            return deferred.promise;
        })
        .then(function (good) {
            if (!good){ return null; }

            logger.info('train', 'read dataset file');
            return read(datasetPath)
                .then(function (dataset) {
                    var svm = new SVM(config);
                    logger.info('train', 'start training');
                    return svm
                        .train(dataset)
                        .progress(function (ratio) {
                            logger.info('train', 'training progress: '+Math.round(ratio*100)+'%');
                        });
                })
                .spread(function (model, report) {
                    var template;
                    switch (config.svmType){
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


                    var bestConf = _o.mixIn({}, config, model.params);
                    // reformat conf a bit
                    _o.forOwn(bestConf, function(value, key){
                        if (key === 'svmType'){
                            bestConf.svmType = svmTypesString[value];
                        }
                        else if (key === 'kernelType'){
                            bestConf.kernelType = kernelTypesString[value];
                        }
                        else {
                            var cKey = _s.camelCase('render-' + key.replace(/_/g, '-'));
                            bestConf[cKey] = req(key, config);
                        }
                    });
                    logger.info('template', 'configuration', {
                        template: 'configuration',
                        json: bestConf
                    });
                    return [model, report];
                })
                .spread(function (model, report) {

                    if (options.pathToModel){
                        return saveModel(model, options.pathToModel).then(function () {
                            logger.info('train', 'model saved @'+options.pathToModel);
                            return [model, report];
                        });
                    }
                    else if (interactive) {
                        // ask user if he wants to save the model somewhere
                        return askIfSaveModel(logger)
                            .then(function (save) {
                                if (save){
                                    return askWhereToSaveIt(logger).then(function (location) {
                                        return saveModel(model, location).then(function () {
                                            logger.info('train', 'model saved @'+location);
                                            return [model, report];
                                        });
                                    });
                                }
                            });
                    }
                    else {
                        return [model, report];
                    }
                });
        });
}

// -------- prompts
function askSvmType(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'svmType',
        'message': 'Classifier',
        'default': svmTypesString[defaultValue || 0],
        'type': 'list',
        'choices': svmTypesString
    }).then(function (answer) {
        return svmTypes[answer];
    });
}
function askKernelType(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'kernelType',
        'message': 'Kernel',
        'default': kernelTypesString[defaultValue || 0],
        'type': 'list',
        'choices': kernelTypesString
    }).then(function (answer) { return kernelTypes[answer]; });
}
function askDegree(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'degree',
        'message': 'Possible values for '+ chalk.green('degree') +' parameter',
        'default': defaultValue,
        'type': 'input',
        'validate': validate
    }).then(function (answer) { return parseNumbers(answer); });
}
function askR(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'r',
        'message': 'Possible values for '+ chalk.green('coef0 (r)') +' parameter',
        'default': _l.toString(defaultValue),
        'type': 'input',
        'validate': validate
    }).then(function (answer) { return parseNumbers(answer); });
}
function askCost(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'c',
        'message': 'Possible values for '+ chalk.green('cost (c)') +' parameter',
        'default': _l.toString(defaultValue),
        'type': 'input',
        'validate': validate
    }).then(function (answer) {
        return parseNumbers(answer);
    });
}
function askNu(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'nu',
        'message': 'Possible values for '+ chalk.green('nu') +' parameter',
        'default': _l.toString(defaultValue),
        'type': 'input',
        'validate': validate
    }).then(function (answer) {
        return parseNumbers(answer);
    });
}
function askEpsilon(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'epsilon',
        'message': 'Possible values for '+ chalk.green('epsilon') +' parameter',
        'default': _l.toString(defaultValue),
        'type': 'input',
        'validate': validate
    }).then(function (answer) { return parseNumbers(answer); });
}
function askGamma(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'gamma',
        'message': 'Possible values for '+ chalk.green('gamma') +' parameter',
        'default': _l.toString(defaultValue),
        'type': 'input',
        'validate': validate
    }).then(function (answer) { return parseNumbers(answer); });
}
function askNormalize(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'normalize',
        'message': 'Normalize dataset',
        'default': defaultValue,
        'type': 'confirm',
        'validate': validate
    }).then(function(answer){
        if (_l.isBoolean(answer)){ return answer; }
        return _s.typecast(answer);
    });
}
function askReduce(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'reduce',
        'message': 'Reduce dataset using PCA',
        'default': defaultValue,
        'type': 'confirm',
        'validate': validate
    }).then(function(answer){
        if (_l.isBoolean(answer)){ return answer; }
        return _s.typecast(answer);
    });
}
function askRetainedVariance(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'retainedVariance',
        'message': 'Retained variance for PCA reduction',
        'default': defaultValue,
        'type': 'input',
        'validate': validate
    }).then(function(answer){ return _s.typecast(answer); });
}
function askKFold(logger, defaultValue) {
    return Q.nfcall(logger.prompt.bind(logger), {
        'name': 'kFold',
        'message': 'k parameter for k-fold cross-validation (set k-fold=1 to disable CV)',
        'default': defaultValue,
        'type': 'input',
        'validate': validate
    }).then(function(answer){ return _s.typecast(answer); });
}

function askConfirmation(logger, config){
    var cleanConfig = _o.merge({}, config);
    // Cleanup empty props (null values, empty strings, objects and arrays)
    _o.forOwn(cleanConfig, function (value, key) {
        if (key === 'svmType'){
            cleanConfig.svmType = svmTypesString[value];
        }
        else if (key === 'kernelType'){
            cleanConfig.kernelType = kernelTypesString[value];
        }
        else if (key === 'cwd' ||
            key === 'argv' ||
            key === 'interactive' ||
            key === 'color'){
            delete cleanConfig[key];
        }
        else if (value === null ||
            _l.isEmpty(value) &&
            !_l.isNumber(value) &&
            !_l.isBoolean(value)) {
            delete cleanConfig[key];
        }
    });
    if (!cleanConfig.reduce){
        delete cleanConfig.retainedVariance;
    }
    logger.info('json', 'Configuration', { json: cleanConfig });

    // Confirm the json with the user
    return Q.nfcall(logger.prompt.bind(logger), {
        name: 'confirmation',
        type: 'confirm',
        message: 'Looks good?',
        default: true
    }).then(function(answer){
        if (_l.isBoolean(answer)){ return answer; }
        return _s.typecast(answer);
    });
}

function askIfSaveModel(logger) {
    return Q.nfcall(logger.prompt.bind(logger), {
        name: 'saveModel',
        type: 'confirm',
        message: 'Do you want to save the model somewhere?',
        default: false
    }).then(function(answer){
        if (_l.isBoolean(answer)){ return answer; }
        return _s.typecast(answer);
    });
}
function askWhereToSaveIt(logger) {
    return Q.nfcall(logger.prompt.bind(logger), {
        name: 'modelPath',
        type: 'input',
        message: 'Where?',
        default: './model.json'
    }).then(function(answer){ return _s.typecast(answer); });
}



// ----------- helpers
function checkFileExists(file){
    return Q.promise(function (resolve) {
        fs.exists(file, resolve);
    });
}

function isNumberArray(input){
    if (!_l.isArray(input)){
        return false;
    }
    return _a.every(input, function (i) {
        return _l.isNumber(i);
    });
}

function validate(input){
    return true;
}

function parseNumbers(input){
    if (_l.isNumber(input)){
        return [input];
    }
    else if (isNumberArray(input)){
        return input;
    }
    else {
        return input.toString().split(',').map(function (v) {
            return _s.typecast(v);
        });
    }
}

function req(param, config){
    switch (param) {
        case 'gamma':
            return config.kernelType === kernelTypes.POLY ||
                config.kernelType ===  kernelTypes.RBF ||
                config.kernelType ===  kernelTypes.SIGMOID;
        case 'degree':
            return  config.kernelType === kernelTypes.POLY;
        case 'r':
            return config.kernelType === kernelTypes.POLY ||
                config.kernelType ===  kernelTypes.SIGMOID;
        case 'c':
            return config.svmType === svmTypes.C_SVC ||
                config.svmType === svmTypes.EPSILON_SVR ||
                config.svmType === svmTypes.NU_SVR;
        case 'nu':
            return config.svmType === svmTypes.NU_SVC ||
                config.svmType === svmTypes.NU_SVR||
                config.svmType === svmTypes.ONE_CLASS;
        case 'epsilon':
            return  config.svmType === svmTypes.EPSILON_SVR;
        case 'kFold':
        case 'normalize':
        case 'reduce':
        case 'eps':
        case 'cacheSize':
        case 'shrinking':
        case 'probability':
            return true;
        case 'retainedVariance':
            return config.reduce;
        default :
            return false;
    }
}

function saveModel(model, file) {
    return Q.nfcall(fs.writeFile, file, JSON.stringify(model));
}

// -------------------

train.options = function (argv) {
    var options = cli.readOptions({
        'svm-type': { type: String, shorthand: 'S'},
        'kernel-type': { type: String, shorthand: 'K'},
        'degree': { type: [Number, Array], shorthand: 'D'},
        'r': { type: [Number, Array], shorthand: 'R'},
        'c': { type: [Number, Array], shorthand: 'C'},
        'nu': { type: [Number, Array], shorthand: 'N'},
        'epsilon': { type: [Number, Array], shorthand: 'E'},
        'gamma': { type: [Number, Array], shorthand: 'G'},

        'k-fold': { type: Number },
        'normalize': { type: Boolean },
        'reduce': { type: Boolean },
        'retained-variance': { type: Number },
        'eps': { type: Number },
        'cache-size': { type: Number },
        'shrinking': { type: Boolean },
        'probability': { type: Boolean },
        'color': { type: Boolean },
        'interactive': { type: Boolean, shorthand: 'i' }
    }, argv);
    if (options.svmType){
        options.svmType = svmTypes[options.svmType] || svmTypes.C_SVC;
    }
    if (options.kernelType){
        options.kernelType = kernelTypes[options.kernelType] || kernelTypes.LINEAR;
    }

    return options;
};

train.line = function (logger, argv) {
    var options = train.options(argv);
    if (options.argv.remain[1]){
        options.pathToDataset = options.argv.remain[1];
        if (options.argv.remain[2]){
            options.pathToModel = options.argv.remain[2];
        }
    }
    delete options.argv;
    return train(logger , options);
};

train.completion = function () {
    // TODO
};

module.exports = train;
