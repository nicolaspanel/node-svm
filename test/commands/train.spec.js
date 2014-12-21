'use strict';

var path = require('path');
var expect = require('expect.js');

var helpers = require('../helpers');
var commands = helpers.require('lib/commands');
var defaultConfig = require('../../lib/core/config');


describe('node-svm train', function () {
    var config;
    var defaults;
    beforeEach(function () {
        defaultConfig.reset();
        defaults = defaultConfig();
        config = {
            interactive: true,
            pathToDataset: './examples/datasets/xor.json',
            kFold: 1
        };
    });

    it('should ask for cost and gamma when using C_SVC with RBF kernel', function (done) {
        var asked = {};
        var logger = commands.train(config);
        logger.on('prompt', function(prompts, answer){
            var answers = {};
            prompts.forEach(function (prompt) {
                asked[prompt.name] = (asked[prompt.name] || 0) +1;
                switch (prompt.name){
                    case 'svmType':
                        answers.svmType = 'C_SVC';
                        break ;
                    case 'kernelType':
                        answers.kernelType = 'RBF';
                        break ;
                    case 'confirmation':
                        answers.confirmation = 'true';
                        break;
                    case 'saveModel':
                        answers.saveModel = 'false';
                        break;
                    default :
                        answers[prompt.name] = defaults[prompt.name].toString();
                }
            });
            answer(answers);
        });

        logger.on('end', function(){
            expect(asked).to.eql({
                svmType: 1,
                kernelType:1,
                c: 1,
                gamma: 1,
                normalize: 1,
                reduce: 1,
                retainedVariance: 1,
                confirmation: 1,
                saveModel: 1
            });
            done();
        });

    });

    it('should ask for cost and nu when using NU_SVR with LINEAR kernel', function (done) {
        var asked = {};
        var logger = commands.train(config);
        logger.on('prompt', function(prompts, answer){
            var answers = {};
            prompts.forEach(function (prompt) {
                asked[prompt.name] = (asked[prompt.name] || 0) +1;
                switch (prompt.name){
                    case 'svmType':
                        answers.svmType = 'NU_SVR';
                        break ;
                    case 'kernelType':
                        answers.kernelType = 'LINEAR';
                        break ;
                    case 'c':
                        answers.c = '1';
                        break;
                    case 'nu':
                        answers.nu = '0.5';
                        break;
                    case 'confirmation':
                        answers.confirmation = 'true';
                        break;
                    case 'saveModel':
                        answers.saveModel = 'false';
                        break;
                    default :
                        answers[prompt.name] = defaults[prompt.name].toString();
                }
            });
            answer(answers);
        });

        logger.on('end', function(){
            expect(asked).to.eql({
                svmType: 1,
                kernelType:1,
                c: 1,
                nu: 1,
                normalize: 1,
                reduce: 1,
                retainedVariance: 1,
                confirmation: 1,
                saveModel: 1
            });
            done();
        });

    });

    it('should ask for nothing if interactions disabled', function (done) {
        config.interactive = false;
        var logger = commands.train(config);
        logger.on('prompt', function(prompts, answer){
            throw new Error('no prompt expected ');
        });

        logger.on('end', function(){
            done();
        });

    });


});