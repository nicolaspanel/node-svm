'use strict';

var expect = require('expect.js');

var helpers = require('../helpers');
var commands = helpers.require('lib/commands');

describe('node-svm evaluate', function () {
    it('should return a perfect accuarcy on xor', function(done){
        var logger = commands.evaluate({
            pathToModel: './examples/models/xor.json',
            pathToTestset:'./examples/datasets/xor.json'
        });
        logger.on('end',function(report){
            expect(report.accuracy).to.be(1);
            expect(report.precision).to.be(1);
            expect(report.recall).to.be(1);
            expect(report.fscore).to.be(1);
            expect(report.size).to.be(4);
            done();
        });
    });
    it('should support both normalization and reduction', function(done){
        var logger = commands.evaluate({
            pathToModel: './examples/models/svmguide1.json',
            pathToTestset:'./examples/datasets/svmguide1.t.ds'
        });
        logger.on('end',function(report){
            expect(report.accuracy).to.be.greaterThan(0.95);
            done();
        });
    });
    it('should work without normalization and without reduction', function(done){
        var logger = commands.evaluate({
            pathToModel: './examples/models/svmguide1--no-normalize--no-reduce.json',
            pathToTestset:'./examples/datasets/svmguide1.t.ds'
        });
        logger.on('end',function(report){
            expect(report.accuracy).to.be(0.66925); // see http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, p.9.
            done();
        });
    });
    it('should work with normalization and no reduction', function(done){
        var logger = commands.evaluate({
            pathToModel: './examples/models/svmguide1--no-reduce.json',
            pathToTestset:'./examples/datasets/svmguide1.t.ds'
        });
        logger.on('end',function(report){
            expect(report.accuracy).to.be.greaterThan(0.95); // Normalization make a significant diff!
            done();
        });
    });
    it('should work with reduction and no normalization', function(done){
        var logger = commands.evaluate({
            pathToModel: './examples/models/svmguide1--no-normalize.json',
            pathToTestset:'./examples/datasets/svmguide1.t.ds'
        });
        logger.on('end',function(report){
            expect(report.accuracy).to.be.greaterThan(0.65);
            done();
        });
    });

});