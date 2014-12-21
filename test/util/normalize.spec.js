'use strict';

var utils = require('../../lib/util');
var expect = require('expect.js');
var xorProblem = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
];

var xorNormProblem = [
    [[-1, -1], 0],
    [[-1,  1], 1],
    [[ 1, -1], 1],
    [[ 1,  1], 0]
];

describe('#normalizeDataset', function(){

    it('should be able to normalize the xor problem', function () {
        var result = utils.normalizeDataset(xorProblem);
        expect(result.mu).to.eql([0.5, 0.5]);
        expect(result.sigma).to.eql([0.5,0.5]);
        expect(result.dataset).to.eql(xorNormProblem);
    });

    it('should be able to normalize an already normalized problem', function() {
        var result = utils.normalizeDataset(xorNormProblem);
        expect(result.dataset).to.eql(xorNormProblem);
    });

    it('should be able to normalize the xor problem with custom mu and sigma', function () {
        var result = utils.normalizeDataset(xorProblem, [0, 0], [1, 1]);
        expect(result.dataset).to.eql(xorProblem); // no changes
    });
});