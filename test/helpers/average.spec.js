'use strict';

var helpers = require('../../lib/helpers');
var expect = require('expect.js');

describe('#average', function(){

    it('work with 1d array', function () {
        expect(helpers.avg([0,1,2,3,4])).to.be(2);
    });

});