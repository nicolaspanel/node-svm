'use strict';

var assert = require('assert');
var mout = require('mout'),
    _a = mout.array;
var numeric = require('numeric');

var avg = require('./average');
var std = require('./standard-deviation');


function normalizeInput(input, mu, sigma){
    assert(input instanceof Array, 'input must be a 1d array');
    assert(mu instanceof Array, 'mu must be a 1d array');
    assert(sigma instanceof Array, 'sigma must be a 1d array');
    var sigmaInv = sigma.map(function(value){ return value === 0 ? 1 : 1 / value;});
    return numeric.mul(numeric.add(input, numeric.neg(mu)), sigmaInv);
}

function normalizeDataSet(dataset, mu, sigma){

    assert(dataset instanceof Array, 'dataset must be an list of [X,y] tuples');
    assert(dataset.length>0, 'dataset cannot be empty');

    var X = dataset.map(function(ex){return ex[0];}),
        n = numeric.dim(X)[0] || 0,
        m = numeric.dim(X)[1] || 0;

    assert(m>0, 'number of features must be gt 0');

    mu = mu || _a.range(m-1).map(function(i){
        return avg(X.map(function(x){ return x[i] || 0; }));
    });
    sigma = sigma || _a.range(m-1).map(function(i){
        return std(X.map(function(x){ return x[i] || 0; }));
    });

    return {
        dataset: dataset.map(function(l){ return [normalizeInput(l[0], mu, sigma), l[1]]; }),
        mu: mu,
        sigma: sigma
    };
}

module.exports = {
    normalizeInput: normalizeInput,
    normalizeDataSet: normalizeDataSet
};