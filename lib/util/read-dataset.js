'use strict';

var Q = require('q');
var fs = require('fs');
var path = require('path');
var _a = require('mout/array');
var _ = require('underscore');
var numeric = require('numeric');


var readLibsvm = function (data) {
    var dataset = [];
    data = data.toString();
    var lines = _.chain(data.split('\n'))
        .filter(function (str) { return str.trim() !== '';  }) // remove empty lines
        .map(function(line){
            var elts = line.split(' ');
            return {
                y: elts[0],
                features: _.chain(elts).rest(1).map(function(str){
                    var node = str.split(':');
                    return {
                        index: parseInt(node[0], 10),
                        value: parseFloat(node[1])
                    };
                }).value()
            };
        });
    var nbFeatures = lines
        .pluck('features')
        .flatten()
        .pluck('index')
        .max()
        .value();
    console.log(nbFeatures);

    return lines
        .map(function (line) {
            var x = numeric.rep([nbFeatures],0);
            line.features.forEach(function(f){
                x[f.index-1]= f.value;
            });
            return [x, line.y];
        }).value();
};

var readJson = function (data) {
  return JSON.parse(data.toString());
};

var read = function(fileName){
    var deferred = Q.defer();
    return Q.nfcall(fs.readFile, fileName)
        .then(function (data) {
            switch (path.extname(fileName)){
                case '.json':
                    return readJson(data);
                default :
                    return readLibsvm(data);
            }
        });
};

module.exports = read;