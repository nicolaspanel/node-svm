'use strict';

var Q = require('q'),
    Logger = require('../util/logger');


/**
 * Require commands only when called.
 *
 * Running `commandFactory(id)` is equivalent to `require(id)`. Both calls return
 * a command function. The difference is that `cmd = commandFactory()` and `cmd()`
 * return as soon as possible and load and execute the command asynchronously.
 */
function commandFactory(id) {
    if (process.env.STRICT_REQUIRE) {
        require(id);
    }

    function command() {
        var commandArgs = [].slice.call(arguments);

        return withLogger(function (logger) {
            commandArgs.unshift(logger);
            return require(id).apply(undefined, commandArgs);
        });
    }

    function runFromArgv(argv) {
        return withLogger(function (logger) {
            return require(id).line.call(undefined, logger, argv);
        });
    }

    function withLogger(func) {
        var logger = new Logger();

        Q.try(func, logger)
            .done(function () {
                var args = [].slice.call(arguments);
                args.unshift('end');
                logger.emit.apply(logger, args);
            }, function (error) {
                console.log(error);
                logger.emit('error', error);
            });

        return logger;
    }

    command.line = runFromArgv;
    return command;
}


module.exports = {
//    completion: commandFactory('./completion'),
    help: commandFactory('./help'),
    train: commandFactory('./train'),
    evaluate: commandFactory('./evaluate')
};