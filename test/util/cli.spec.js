'use strict';

var expect = require('expect.js');
var utils = require('../../lib/util');


describe('cli', function () {
    describe('help', function () {
        var opts = { help: { type: Boolean, shorthand: 'h' }};
        it('should support shorthand', function () {
            var args = 'node file.js -h'.split(' '),
                options = utils.cli.readOptions(opts, args);

            expect(options).to.have.property('help', true);
        });
        it('should support flag', function () {
            var args = 'node file.js --help'.split(' '),
                options = utils.cli.readOptions(opts, args);

            expect(options).to.have.property('help', true);
        });
        it('should support command', function () {
            var args ='node file.js help'.split(' '),
                options = utils.cli.readOptions(opts, args);

            expect(options.argv.remain).to.contain('help');
        });
        it('should support help for a given cmd', function () {
            var args ='node file.js cmd -h'.split(' '),
                options = utils.cli.readOptions(opts, args);

            expect(options.argv.remain).to.contain('cmd');
            expect(options).to.have.property('help', true);
        });
    });
    describe('version', function () {
        var opts = { version: { type: Boolean, shorthand: 'v' }};
        it('should support shorthand', function () {
            var args = ['node', 'path', '-v'],
                options = utils.cli.readOptions(opts, args);

            expect(options).to.have.property('version', true);
        });
        it('should support flag', function () {
            var args = ['node', 'path', '--version'],
                options = utils.cli.readOptions(opts, args);

            expect(options).to.have.property('version', true);
        });
        it('should support command', function () {
            var args = ['node', 'path', 'version'],
                options = utils.cli.readOptions(opts, args);

            expect(options.argv.remain).to.contain('version');
        });
    });

});
