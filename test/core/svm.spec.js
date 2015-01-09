'use strict';

var expect = require('expect.js');
var Q = require('q');
var BaseSVM = require('../../lib/core/base-svm');
var SVM = require('../../lib/core/svm');
var svmTypes = require('../../lib/core/svm-types');
var kernelTypes = require('../../lib/core/kernel-types');

var xor = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];
var xorRegression = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];
var redundantXor = [
    [[0, 0, 0, 0], 0],
    [[0, 0, 1, 1], 1],
    [[1, 1, 0, 0], 1],
    [[1, 1, 1, 1], 0]
];

describe('baseSVM', function () {
    var baseSvm;
    beforeEach(function () {
        baseSvm = new BaseSVM();
    });

    it('should not be trained', function () {
        expect(baseSvm.isTrained()).to.be(false);
    });

    it('can be trained', function (done) {
        baseSvm.train(xor)
            .then(function (model) {
                expect(model).to.be.an('object');
                expect(baseSvm.isTrained()).to.be(true);
            }).done(function(){
                done();
            });
    });
    describe('once trained', function () {
        var trainedModel;
        beforeEach(function (done) {
            baseSvm
            .train(xor, { 
                svmType: svmTypes.C_SVC,
                kernelType: kernelTypes.RBF,
                c: 1,
                gamma: 0.5 ,
                probability: true
            })
            .then(function (model) {
                trainedModel = model;
                done();
            });
        });
        it('can predict synchronously', function () {
            xor.forEach(function (ex) {
                expect(baseSvm.predictSync(ex[0])).to.be(ex[1]);
            });
        });
        it('can predict probability synchronously', function () {
            xor.forEach(function (ex) {
                var props = baseSvm.predictProbabilitiesSync(ex[0]);
                expect(props).to.have.property('0');
                expect(props).to.have.property('1');
            });
        });
        it('can predict probability asynchronously', function (done) {
            Q.all(xor.map(function(ex){
                return baseSvm.predictProbabilities(ex[0]).then(function(props){
                    expect(props).to.have.property('0');
                    expect(props).to.have.property('1');
                });
            })).done(function () {
                done();
            });
        });
        it('can predict asynchronously', function (done) {
            Q.all(xor.map(function(ex){
                return baseSvm.predict(ex[0]).then(function(predicted){
                    expect(predicted).to.be(ex[1]);
                });
            })).done(function () {
                done();
            });

        });
        describe('model', function () {
            it('should contain SV', function(){
                expect(trainedModel.supportVectors).to.be.an('array');
                expect(trainedModel.nbSupportVectors).to.be.an('array');
            });
            it('should contain labels', function(){
                expect(trainedModel.labels).to.be.an('array');
                expect(trainedModel.labels).to.eql([0, 1]);
            });
            it('should contain params', function(){
                expect(trainedModel.params).to.eql({
                    svmType: svmTypes.C_SVC,
                    kernelType: kernelTypes.RBF,
                    c: 1,
                    gamma: 0.5,

                    cacheSize: 100,
                    eps: 1e-3,
                    shrinking: true,
                    probability: true
                });
            });
        });

        describe('once restored', function () {
            var restored = null;
            beforeEach(function () {
                restored = BaseSVM.restore(trainedModel);
            });
            it('can be restore from model and used for new predictions', function(){
                xor.forEach(function (ex) {
                    expect(restored.predictSync(ex[0])).to.be(ex[1]);
                });
            });
            it('can be restore from model and used for new probabilities predictions', function(){
                xor.forEach(function (ex) {
                    var props = restored.predictProbabilitiesSync(ex[0]);
                    expect(props).to.have.property('0');
                    expect(props).to.have.property('1');
                });
            });
        });
    });

});

describe('SVM', function(){
    describe('using default config', function () {
        var svm;
        beforeEach(function () {
            svm = new SVM({ kFold:1 });
        });

        it('should use C_SVC classifier', function(){
            expect(svm.getSvmType()).to.be(svmTypes.C_SVC);
        });
        it('should use RBF kernel', function(){
            expect(svm.getKernelType()).to.be(kernelTypes.RBF);
        });
        it('should normalise inputs', function(){
            expect(svm.normalize()).to.be(true);
        });
        it('should reduce inputs', function(){
            expect(svm.reduce()).to.be(true);
        });
        it('should not be trained already', function(){
            expect(svm.isTrained()).to.be(false);
        });
        it('can be trained', function(done){
            svm.train(xor)
                .spread(function(){
                    expect(svm.isTrained()).to.be(true);
                    expect(svm.isTraining()).to.be(false);
                })
                .fail(function (err) {
                    throw err;
                })
                .done(done);
        });

        describe('once trained', function () {
            var trainedModel, trainingReport;
            beforeEach(function (done) {
                svm.train(xor)
                    .spread(function(model, report){
                        trainedModel = model;
                        trainingReport = report;
                    })
                    .fail(function (err) {
                        throw err;
                    })
                    .done(done);
            });

            it('should generate a training report', function () {
                expect(trainingReport).to.be.an('object');
                expect(trainingReport.retainedVariance).to.be(1);
            });
            it('can be evaluated against a test set', function () {
                var report = svm.evaluate(xor);
                expect(report.fscore).to.be(1);
            });

            it('should generate a model containing retained parameters', function () {
                expect(trainedModel).to.be.an('object');
                expect(trainedModel.labels).to.eql([0,1]);
                expect(trainedModel.nrClass).to.be(2);
                expect(trainedModel.l).to.be(xor.length);
                expect(trainedModel.supportVectors).to.be.an('array');
                expect(trainedModel.params).to.be.an('object');
                expect(trainedModel.params.mu).to.eql([0.5, 0.5]);
                expect(trainedModel.params.sigma).to.eql([0.5, 0.5]);
            });

            describe('and trained model used to create new classifiers', function () {
                var newSvm;
                beforeEach(function () {
                    newSvm = new SVM({}, trainedModel);
                    expect(newSvm.isTrained()).to.be(true);
                });
                it('can predict synchronously', function () {
                    xor.forEach(function (ex) {
                        expect(newSvm.predictSync(ex[0])).to.be(ex[1]);
                    });
                });
                it('can predict asynchronously', function (done) {
                    Q.all(xor.map(function(ex){
                        return newSvm.predict(ex[0]).then(function(predicted){
                            expect(predicted).to.be(ex[1]);
                        });
                    })).done(function () {
                        done();
                    });

                });
                it('can be trained again', function (done) {
                    newSvm.train(xor)
                        .spread(function(){
                            expect(svm.isTraining()).to.be(false);
                        })
                        .fail(function (err) {
                            throw err;
                        })
                        .done(done);

                });
            });

        });
    });

    describe('using EPSILON_SVR', function () {
        var svm;
        beforeEach(function () {
            svm = new SVM({
                svmType: svmTypes.EPSILON_SVR,
                kFold:1
            });
        });
        it('should use EPSILON_SVR classifier', function(){
            expect(svm.getSvmType()).to.be(svmTypes.EPSILON_SVR);
        });

        describe('once trained', function () {
            var trainedModel, trainingReport;
            beforeEach(function (done) {
                svm.train(xor)
                    .spread(function(model, report){
                        trainedModel = model;
                        trainingReport = report;
                    })
                    .fail(function (err) {
                        throw err;
                    })
                    .done(done);
            });

            it('should generate a training report', function () {
                expect(trainingReport).to.be.an('object');
                expect(trainingReport.retainedVariance).to.be(1);
            });
            it('can be evaluated against a test set', function () {
                var report = svm.evaluate(xor);
                expect(report.mse).to.be.a('number');
                expect(report.mse).to.be.lessThan(0.5);
                expect(report.mean).to.be.lessThan(1e-3);
            });

            it('should generate a model containing retained parameters', function () {
                expect(trainedModel).to.be.an('object');
                expect(trainedModel.l).to.be(xor.length);
                expect(trainedModel.supportVectors).to.be.an('array');
                expect(trainedModel.params).to.be.an('object');
            });

        });
    });

    describe('with kFold set to 1 (i.e. using entire dataset for both training and tests)', function () {
        var svm;
        beforeEach(function () {
            svm = new SVM({
                kFold: 1
            });
        });

        it('should report an fscore of 1', function(done){
            svm.train(xor)
                .spread(function(model, report){
                    expect(report.fscore).to.be(1);
                })
                .fail(function (err) {
                    throw err;
                })
                .done(done);
        });
    });

    describe('with normalization disabled', function () {
        var svm;
        beforeEach(function () {
            svm = new SVM({
                normalize: false
            });
        });

        it('should have mu set to 0 and sigma set to 1', function(done){
            svm.train(xor)
                .spread(function(model, report){
                    expect(model.params.mu).to.eql([0,0]);
                    expect(model.params.sigma).to.eql([1,1]);
                })
                .fail(function (err) {
                    throw err;
                })
                .done(done);
        });
    });

    describe('using PCA reduction on a highly redundant dataset', function () {
        var svm;
        beforeEach(function () {
            svm = new SVM({
                reduce: true // Default
            });
        });

        it('should have mu set to 0 and sigma set to 1', function(done){
            svm.train(redundantXor)
                .spread(function(model, report){
                    expect(report.retainedVariance).to.be(1);
                })
                .fail(function (err) {
                    throw err;
                })
                .done(done);
        });
    });

});
