'use strict';

var split = require('../../lib/util/split-dataset');
var expect = require('expect.js');

describe('#split-dataset', function(){

    it('should return 3 subsets if kFold is 3', function () {
        var dataset = [0,1,2,3,4],
            result = split(dataset, 3);
        expect(result).to.be.an('array');
        expect(result.length).to.be(3);

        result.forEach(function (r) {
            expect(r.train).to.be.an('array');
            expect(r.test).to.be.an('array');
        });
    });
    it('should return the entire dataset for both training and testing if kFold set to 1', function () {
        var dataset = [0,1,2,3,4],
            result = split(dataset, 1);

        expect(result).to.be.an('array');
        expect(result.length).to.be(1);
        result.forEach(function (r) {
            expect(r.train).to.be.an('array');
            expect(r.train.length).to.be(5);
            expect(r.test).to.be.an('array');
            expect(r.test.length).to.be(5);
        });
    });

});