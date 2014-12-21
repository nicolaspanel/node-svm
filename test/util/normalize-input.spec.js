'use strict';

var ni = require('../../lib/util/normalize-input');
var expect = require('expect.js');

describe('#normalizeInput', function(){
    it('should be able to normalize 2d inputs with custom mu and sigma', function () {
        var result = ni([1, 1], [0, 0], [1, 1]);
        expect(result).to.eql([1,1]); // no changes
    });
});