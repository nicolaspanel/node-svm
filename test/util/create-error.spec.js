'use strict';

var expect = require('expect.js'),
    createError = require('../../lib/util/create-error');

describe('create-error', function () {

    it('should accept msg and code', function () {
        var err = createError('msg', 'ECODE');
        expect(err).to.be.an(Error);
        expect(err.code).to.be('ECODE');
    });

    it('should accept other props', function () {
        var err = createError('msg', 'ECODE', {foo: 'bar'});
        expect(err.foo).to.be('bar');
    });
});