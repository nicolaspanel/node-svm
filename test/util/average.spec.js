'use strict';

var utils = require('../../lib/util');
var expect = require('expect.js');

describe('#average', function(){

    it('work with 1d array', function () {
        expect(utils.avg([0,1,2,3,4])).to.be(2);
    });

});