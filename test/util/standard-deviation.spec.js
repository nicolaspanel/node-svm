'use strict';

var std = require('../../lib/util/standard-deviation');
var expect = require('expect.js');
var _a = require('mout/array');

describe('#std', function(){

    it('work with 1d array', function () {
        expect(std([0,1,2,3,4])).to.be(Math.pow(2, 0.5));
    });

});