'use strict';

var utils = require('../../lib/util');
var expect = require('expect.js');

describe('#cross-combinations', function(){

    it('should work', function () {
        var combs = utils.crossCombinations([
            [0,1],
            [0,1],
            []
        ]);
        expect(combs).to.be.an('array');
        expect(combs).to.eql([
            [0,0, null],
            [1,0, null],
            [0,1, null],
            [1,1, null]
        ]);
    });

});