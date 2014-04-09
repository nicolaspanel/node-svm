'use strict';

var nodesvm = require('../lib/nodesvm'),
    datasetFileName = './examples/datasets/housing.ds',
    modelFileName = './examples/models/housing.model';

var svm = new nodesvm.EpsilonSVR({
  kernelType: nodesvm.KernelTypes.RBF,
  gamma: 0.5,
  C: 32,
  epsilon: 0.125,
  normalize: true, // default value
  reduce: true, // default value
  retainedVariance: 0.98
});

svm.once('trained', function  () {
  svm.saveToFile(modelFileName);
});
svm.trainFromFile(datasetFileName);