
'use strict';

var expect = require('expect.js');
var configBuilder = require('../../lib/core/config');
var svmTypes = require('../../lib/core/svm-types');
var kernelTypes = require('../../lib/core/kernel-types');

describe('Config', function () {

    describe('using default options', function () {
        var config;
        beforeEach(function () {
            config = configBuilder();
            // note: take into account ../.node-svmrc file
        });

        it('should return default classifer values', function(){
            expect(config.svmType).to.be(svmTypes.C_SVC);
            expect(config.kernelType).to.be(kernelTypes.RBF);
            expect(config.degree).to.eql([]);
            expect(config.gamma).to.eql([ 0.001, 0.01, 0.5 ]);
            expect(config.r).to.eql([]);
            expect(config.c).to.eql([ 0.01, 0.125, 0.5, 1, 2 ]);
            expect(config.nu).to.eql([]);
            expect(config.epsilon).to.eql([]);
        });

        it('should return default training values', function(){
            expect(config.normalize).to.be(true);
            expect(config.reduce).to.be(true);
            expect(config.retainedVariance).to.be(0.99);
            expect(config.eps).to.be(1e-3);
            expect(config.probability).to.be(false);
        });
        it('should return cli values', function(){
            expect(config.color).to.be(true);
        });
        it('should handle .node-svmrc values', function(){
            expect(config.kFold).to.be(1);
            expect(config.cacheSize).to.be(1000);
        });
    });

    describe('using NU_SVC classifier', function () {
        var config;
        beforeEach(function () {
            config = configBuilder({
                svmType: svmTypes.NU_SVC
            });
            // note: take into account ../.node-svmrc file
        });

        it('should return default classifier values', function(){
            expect(config.svmType).to.be(svmTypes.NU_SVC);
            expect(config.kernelType).to.be(kernelTypes.RBF);
            expect(config.degree).to.eql([]);
            expect(config.gamma).to.eql([ 0.001, 0.01, 0.5 ]);
            expect(config.r).to.eql([]);
            expect(config.c).to.eql([]);
            expect(config.nu).to.eql([ 0.01, 0.125, 0.5, 1 ]);
            expect(config.epsilon).to.eql([]);
        });

    });
    describe('using EPSILON_SVR classifier', function () {
        var config;
        beforeEach(function () {
            config = configBuilder({
                svmType: svmTypes.EPSILON_SVR
            });
            // note: take into account ../.node-svmrc file
        });

        it('should return default classifier values', function(){
            expect(config.svmType).to.be(svmTypes.EPSILON_SVR);
            expect(config.kernelType).to.be(kernelTypes.RBF);
            expect(config.degree).to.eql([]);
            expect(config.gamma).to.eql([ 0.001, 0.01, 0.5 ]);
            expect(config.r).to.eql([]);
            expect(config.c).to.eql([ 0.01, 0.125, 0.5, 1, 2 ]);
            expect(config.nu).to.eql([]);
            expect(config.epsilon).to.eql([ 0.01, 0.125, 0.5, 1 ]);
        });

    });

    describe('using NU_SVR classifier', function () {
        var config;
        beforeEach(function () {
            config = configBuilder({
                svmType: svmTypes.NU_SVR
            });
            // note: take into account ../.node-svmrc file
        });

        it('should return default classifier values', function(){
            expect(config.svmType).to.be(svmTypes.NU_SVR);
            expect(config.kernelType).to.be(kernelTypes.RBF);
            expect(config.degree).to.eql([]);
            expect(config.gamma).to.eql([ 0.001, 0.01, 0.5 ]);
            expect(config.r).to.eql([]);
            expect(config.c).to.eql([ 0.01, 0.125, 0.5, 1, 2 ]);
            expect(config.nu).to.eql([ 0.01, 0.125, 0.5, 1 ]);
            expect(config.epsilon).to.eql([]);
        });

    });

    describe('using LINEAR kernel', function () {
        var config;
        beforeEach(function () {
            config = configBuilder({
                kernelType: kernelTypes.LINEAR
            });
            // note: take into account ../.node-svmrc file
        });

        it('should return default values for this given kernel', function(){
            expect(config.svmType).to.be(svmTypes.C_SVC);
            expect(config.kernelType).to.be(kernelTypes.LINEAR);
            expect(config.degree).to.eql([]);
            expect(config.gamma).to.eql([]);
            expect(config.r).to.eql([]);
            expect(config.c).to.eql([ 0.01, 0.125, 0.5, 1, 2 ]);
            expect(config.nu).to.eql([]);
            expect(config.epsilon).to.eql([]);
        });

    });
    describe('using POLY kernel', function () {
        var config;
        beforeEach(function () {
            config = configBuilder({
                kernelType: kernelTypes.POLY
            });
            // note: take into account ../.node-svmrc file
        });

        it('should return default values for this given kernel', function(){
            expect(config.svmType).to.be(svmTypes.C_SVC);
            expect(config.kernelType).to.be(kernelTypes.POLY);
            expect(config.degree).to.eql([2,3,4]);
            expect(config.gamma).to.eql([ 0.001, 0.01, 0.5 ]);
            expect(config.r).to.eql([0.125, 0.5, 0, 1]);
            expect(config.c).to.eql([ 0.01, 0.125, 0.5, 1, 2 ]);
            expect(config.nu).to.eql([]);
            expect(config.epsilon).to.eql([]);
        });

    });
    describe('using SIGMOID kernel', function () {
        var config;
        beforeEach(function () {
            config = configBuilder({
                kernelType: kernelTypes.SIGMOID
            });
            // note: take into account ../.node-svmrc file
        });

        it('should return default values for this given kernel', function(){
            expect(config.svmType).to.be(svmTypes.C_SVC);
            expect(config.kernelType).to.be(kernelTypes.SIGMOID);
            expect(config.degree).to.eql([]);
            expect(config.gamma).to.eql([ 0.001, 0.01, 0.5 ]);
            expect(config.r).to.eql([0.125, 0.5, 0, 1]);
            expect(config.c).to.eql([ 0.01, 0.125, 0.5, 1, 2 ]);
            expect(config.nu).to.eql([]);
            expect(config.epsilon).to.eql([]);
        });

    });

    describe('using scalar options', function () {
        var config;
        beforeEach(function () {
            config = configBuilder({
                kernelType: kernelTypes.POLY,
                degree: 3,
                gamma: 1e-2,
                r: 0,
                c: 1
            });
            // note: take into account ../.node-svmrc file
        });

        it('should return default values for this given kernel', function(){
            expect(config.svmType).to.be(svmTypes.C_SVC);
            expect(config.kernelType).to.be(kernelTypes.POLY);
            expect(config.degree).to.eql([3]);
            expect(config.gamma).to.eql([0.01]);
            expect(config.r).to.eql([0]);
            expect(config.c).to.eql([1]);
        });
    });
});