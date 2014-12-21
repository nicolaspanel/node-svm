'use strict';

var avg = require('../../lib/util/average');
var expect = require('expect.js');

describe('#average', function(){

    it('work with 1d array', function () {
        expect(avg([0,1,2,3,4])).to.be(2);
    });

});