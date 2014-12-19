'use strict';

var helpers = require('../../lib/helpers');
var expect = require('expect.js');
var xorProblem = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];

describe('#read', function(){
    describe('libsvm format', function () {
        it('should be able to read the xor problem', function (done) {
            helpers.read('./examples/datasets/xor.ds')
                .then(function(problem){
                    expect(problem.length).to.be(4);
                    expect(problem).to.eql(xorProblem);
                }).done(done);
        });
        it('should be able to read the svmguide problem in less than 200ms', function (done) {
            this.timeout(200);
            helpers.read('./examples/datasets/svmguide1.ds')
                .then(function(problem){
                    expect(problem.length).to.be(3089);
                }).done(done);
        });
    });
    describe('json format', function () {
        it('should be able to read the xor problem', function (done) {
            helpers.read('./examples/datasets/xor.json')
                .then(function(problem){
                    expect(problem.length).to.be(4);
                    expect(problem).to.eql(xorProblem);
                }).done(done);
        });

    });
});