'use strict';

var path = require('path');
var expect = require('expect.js');

var helpers = require('../helpers');
var commands = helpers.require('lib/commands');
var svmTypes = helpers.require('lib/core/svm-types');
var kernelTypes = helpers.require('lib/core/kernel-types');
var defaultConfig = require('../../lib/core/config');


describe('node-svm train', function () {
    beforeEach(function () {
        defaultConfig.reset();
    });

    it('should ask for cost and gamma when using C_SVC with RBF kernel', function (done) {
        var defaults = defaultConfig({
            svmType: svmTypes.C_SVC,
            kernelType: kernelTypes.RBF
        });
        var args = {
            svmType: svmTypes.C_SVC,
            kernelType: kernelTypes.RBF,
            interactive: true,
            pathToDataset: './examples/datasets/xor.json'
        };
        var asked = {};
        var logger = commands.train(args);
        logger.on('prompt', function(prompts, answer){
            var answers = {};
            prompts.forEach(function (prompt) {
                asked[prompt.name] = (asked[prompt.name] || 0) +1;
                switch (prompt.name){
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
                c: 1,
                gamma: 1,
                normalize: 1,
                reduce: 1,
                retainedVariance: 1,
                confirmation: 1,
                saveModel: 1,
                kFold: 1
            });
            done();
        });

    });

    it('should ask for nu, degree, r and gamma when using NU_SVC with POLY kernel', function (done) {
        var defaults = defaultConfig({
            svmType: svmTypes.NU_SVC,
            kernelType: kernelTypes.POLY
        });

        var asked = {};
        var logger = commands.train({
            svmType: svmTypes.NU_SVC,
            kernelType: kernelTypes.POLY,
            kFold: 1,
            interactive: true,
            pathToDataset: './examples/datasets/xor.json'
        });

        logger.on('prompt', function(prompts, answer){
            var answers = {};
            prompts.forEach(function (prompt) {
                asked[prompt.name] = (asked[prompt.name] || 0) +1;
                switch (prompt.name){
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
                nu: 1,
                degree: 1,
                gamma: 1,
                r: 1,
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
        var defaults = defaultConfig({
            svmType: svmTypes.NU_SVR,
            kernelType: kernelTypes.LINEAR
        });
        var logger = commands.train({
            svmType: svmTypes.NU_SVR,
            kernelType: kernelTypes.LINEAR,
            reduce: false,
            kFold: 1,
            interactive: true,
            pathToDataset: './examples/datasets/xor.json'
        });
        logger.on('prompt', function(prompts, answer){
            var answers = {};
            prompts.forEach(function (prompt) {
                asked[prompt.name] = (asked[prompt.name] || 0) +1;
                switch (prompt.name){
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
                c: 1,
                nu: 1,
                normalize: 1,
                confirmation: 1,
                saveModel: 1
            });
            done();
        });

    });

    it('should ask for cost, epsilon and gamma when using EPSILON_SVR with RBF kernel', function (done) {
        var asked = {};
        var defaults = defaultConfig({
            svmType: svmTypes.EPSILON_SVR,
            kernelType: kernelTypes.RBF
        });
        var logger = commands.train({
            svmType: svmTypes.EPSILON_SVR,
            kernelType: kernelTypes.RBF,
            reduce: true,
            normalize: false,
            interactive: true,
            kFold: 1,
            pathToDataset: './examples/datasets/xor.json'
        });
        logger.on('prompt', function(prompts, answer){
            var answers = {};
            prompts.forEach(function (prompt) {
                asked[prompt.name] = (asked[prompt.name] || 0) +1;
                switch (prompt.name){
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
                c: 1,
                epsilon: 1,
                gamma: 1,
                retainedVariance: 1,
                confirmation: 1,
                saveModel: 1
            });
            done();
        });

    });

    it('should ask for nothing if interactions disabled', function (done) {
        var logger = commands.train({
            interactive: false,
            pathToDataset: './examples/datasets/xor.json'
        });
        logger.on('prompt', function(prompts, answer){
            throw new Error('no prompt expected ');
        });

        logger.on('end', function(){
            done();
        });

    });

    it('can train ONE_CLASS svm', function(done){
        var defaults = defaultConfig({
            svmType: svmTypes.ONE_CLASS,
            kernelType: kernelTypes.RBF
        });
        var args = {
            interactive: true,
            svmType: svmTypes.ONE_CLASS,
            kernelType: kernelTypes.RBF,
            normalize: false,
            reduce: false,
            nu: 0.1,
            gamma: 0.1,
            pathToDataset: './examples/datasets/one-class.train.json'
        };
        var asked = {};
        var logger = commands.train(args);
        logger.on('prompt', function(prompts, answer){
            var answers = {};
            prompts.forEach(function (prompt) {
                asked[prompt.name] = (asked[prompt.name] || 0) +1;
                switch (prompt.name){
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
                confirmation: 1,
                saveModel: 1,
                kFold: 1
            });
            done();
        });
    });

});