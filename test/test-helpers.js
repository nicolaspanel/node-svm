'use strict';

var Q = require('q');
var path = require('path');
var config = require('../lib/core/config');

beforeEach(function () {
    config.reset();
});

module.exports.require = function(name) {
    return require(path.join(__dirname, '../', name));
};

