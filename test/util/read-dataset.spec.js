'use strict';

var read = require('../../lib/util/read-dataset');
var numeric = require('numeric');
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
            read('./examples/datasets/xor.ds')
                .then(function(problem){
                    expect(problem.length).to.be(4);
                    expect(problem).to.eql(xorProblem);
                }).done(done);
        });
        it('should be able to read the svmguide problem in less than 200ms', function (done) {
            this.timeout(200);
            read('./examples/datasets/svmguide1.ds')
                .then(function(problem){
                    expect(numeric.dim(problem)).to.eql([3089, 2, 4]);
                }).done(done);
        });
        
    });
    describe('json format', function () {
        it('should be able to read the xor problem', function (done) {
            read('./examples/datasets/xor.json')
                .then(function(problem){
                    expect(problem.length).to.be(4);
                    expect(problem).to.eql(xorProblem);
                }).done(done);
        });

    });
});