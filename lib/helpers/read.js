'use strict';

var Q = require('q');
var fs = require('fs');
var path = require('path');
var _a = require('mout/array');
var numeric = require('numeric');


var readLibsvm = function (data) {
    var dataset = [], nbFeatures = 0;
    var lines = _a.filter(data.toString().split('\n'), function(str){
        return str.trim() !== '';  // remove empty lines
    });

    lines.forEach(function(line){
        var elts = line.split(' ');
        for (var i = 1; i < elts.length; i++){
            var node = elts[i].split(':');
            var index = parseInt(node[0], 10);
            if (index > nbFeatures){
                nbFeatures = index;
            }
        }
    });

    lines.forEach(function(line){
        var elts = line.split(' ');
        var node = [];
        node[0] = numeric.rep([nbFeatures],0);
        for (var i = 1; i < elts.length ; i++){
            var indexValue = elts[i].split(':');
            var index = parseInt(indexValue[0], 10);
            node[0][index - 1] = parseFloat(indexValue[1]);
        }
        node[1] = parseFloat(elts[0]);
        dataset.push(node);
    });
    return dataset;
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