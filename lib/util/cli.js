'use strict';

var mout = require('mout'),
    nopt = require('nopt');

function readOptions(options, args) {
    var types = {}, noptOptions, parsedOptions = {}, shorthands = {};

    if (Array.isArray(options)) {
        args = options;
        options = {};
    } else {
        options = options || {};
    }

    mout.object.forOwn(options, function (option, name) {
        types[name] = option.type;
    });
    mout.object.forOwn(options, function (option, name) {
        shorthands[option.shorthand] = '--' + name;
    });
    noptOptions = nopt(types, shorthands, args);

    // Filter only the specified options because nopt parses every --
    // Also make them camel case
    mout.object.forOwn(noptOptions, function (value, key) {
        if (options[key]) {
            parsedOptions[mout.string.camelCase(key)] = value;
        }
    });

    parsedOptions.argv = noptOptions.argv;

    return parsedOptions;
}


module.exports.readOptions = readOptions;