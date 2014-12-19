'use strict';

module.exports = {
    read: require('./read'),
    normalizeDataset : require('./normalize').normalizeDataSet,
    normalizeInput : require('./normalize').normalizeInput,
    avg : require('./average'),
    std : require('./standard-deviation'),
    crossCombinations : require('./cross-combinations'),
    reduce : require('./reduce'),
    splitDataSet: require('./split-data-set')
};