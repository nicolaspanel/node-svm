'use strict';

module.exports = {
    avg : require('./average'),
    cli : require('./cli'),
    createError : require('./create-error'),
    crossCombinations : require('./cross-combinations'),
    Logger: require('./logger'),
    normalizeDataset : require('./normalize').normalizeDataSet,
    normalizeInput : require('./normalize').normalizeInput,
    paths: require('./paths'),
    read: require('./read'),
    readConfig: require('./read-config'),
    reduce : require('./reduce'),
    splitDataSet: require('./split-data-set'),
    std : require('./standard-deviation'),
    template: require('./template')
};