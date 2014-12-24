'use strict';

var assert = require('assert'),
    expect = require('expect.js'),
    Q = require('q'),
    mout = require('mout'),
    svmTypes = require('../../lib/core/svm-types'),
    evaluators = require('../../lib/evaluators'),
    regression = evaluators.regression;

var xor = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];


describe ('Regression Evaluator', function(){
    it('should be default classifier for EPSILON_SVR', function () {
        expect(evaluators.getDefault({ svmType: svmTypes.EPSILON_SVR })).to.be(regression);
    });
    it('should be default classifier for NU_SVR', function () {
        expect(evaluators.getDefault({ svmType: svmTypes.NU_SVR })).to.be(regression);
    });
    describe ('with bad classifier', function () {
        var clf;
        beforeEach(function () {
            clf = { predictSync: function(state){ return 0; } };
        });

        it ('should report bad results', function(){
            var report = regression.evaluate(xor, clf);
            expect(report).to.eql({
                mse: 0.5,
                std: 0.5,
                mean: -0.5,
                size: xor.length
            });
        });

    });

    describe ('with perfect classifier', function(){
        var clf;
        beforeEach(function () {
            clf = {
                predictSync: function(state){
                    return [[0, 1], [1, 0]][state[0]][state[1]];
                }
            };
        });

        it ('should report a mse/std error/mean error of 0', function(){
            var report = regression.evaluate(xor, clf);
            expect(report).to.eql({
                mse: 0,
                std:0,
                mean: 0,
                size: xor.length
            });
        });
    });
});
