'use strict';

var expect = require('expect.js');
var numeric = require('numeric');
var reduce = require('../../lib/util/reduce-dataset');


describe('#reduce', function(){
    describe('with highly redondant dataset', function () {
        var dataset = [
            [[ 0,  0,  0], 0],
            [[ 0,  0,  1], 1],
            [[ 0,  0,  0], 1],
            [[ 0,  0,  1], 0],
            [[ 1,  1,  0], 0],
            [[ 1,  1,  1], 1],
            [[ 1,  1,  0], 1],
            [[ 1,  1,  1], 0]
        ];
        it('should retain 100 percent of the variance', function () {
            var result = reduce(dataset);
            expect(result.retainedVariance).to.be(1);
        });

        it('should reduce inputs to have a dimension of 2', function () {
            var result = reduce(dataset);
            expect(numeric.dim(result.dataset)).to.eql([8, 2, 2]);
            expect(numeric.dim(result.U)).to.eql([3, 2]);
        });
    });
    describe('with non redondant dataset', function(){
        var dataset = [
            [[ 0,  0,  0], 0],
            [[ 0,  0,  1], 1],
            [[ 0,  1,  0], 1],
            [[ 0,  1,  1], 0],
            [[ 1,  0,  0], 0],
            [[ 1,  0,  1], 1],
            [[ 1,  1,  0], 1],
            [[ 1,  1,  1], 0]
        ];

        it('should NOT have been reduced if expect 99% of the variance to be retained', function () {
            var result = reduce(dataset, 0.99);
            expect(numeric.dim(dataset)).to.eql([8, 2, 3]);
            expect(result.retainedVariance).to.be(1);
        });

    });

});