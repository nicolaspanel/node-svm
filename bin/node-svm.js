#!/usr/bin/env node
'use strict';

process.bin = process.title = 'node-svm';

var mout= require('mout'),
    Q = require('q'),
    Logger = require('../lib/util/logger'),
    abbrev = require('abbrev'),
    pkg = require('../package.json'),
    cli = require('../lib/util/cli'),
    commands = require('../lib/commands'),
    renderers = require('../lib/renderers'),
    config = require('../lib/core/config')();


var options, command, commandFunc, logger,
    renderer, logLevel, levels = Logger.LEVELS,
    abbreviations = abbrev(expandNames(commands));


options = cli.readOptions({
    version: { type: Boolean, shorthand: 'v' },
    help: { type: Boolean, shorthand: 'h' }
}, process.argv);


// -- Handle print of version
if (options.version) {
    process.stdout.write(pkg.version + '\n');
    process.exit();
}

// Set logLevel
if (config.silent) {
    logLevel = levels.error;
} else if (config.verbose) {
    logLevel = -Infinity;
    Q.longStackSupport = true;
} else if (config.quiet) {
    logLevel = levels.warn;
} else {
    logLevel = levels[config.logLevel] || levels.info;
}

// Get the command to execute
while (options.argv.remain.length) {
    command = options.argv.remain.join(' ');

    // Alias lookup
    if (abbreviations[command]) {
        command = abbreviations[command].replace(/\s/g, '.');
        break;
    }

    command = command.replace(/\s/g, '.');

    // Direct lookup
    if (mout.object.has(commands, command)) {
        break;
    }

    options.argv.remain.pop();
}

// Execute the command
commandFunc = command && mout.object.get(commands, command);
command = command && command.replace(/\./g, ' ');

// If no command was specified, show npip help
// Do the same if the command is unknown
if (!commandFunc) {
    logger = commands.help();
    command = 'help';
// If the user requested help, show the command's help
// Do the same if the actual command is a group of other commands
} else if (options.help || !commandFunc.line) {
    logger = commands.help(command);
    command = 'help';
// Call the line method
} else {
    logger = commandFunc.line(process.argv);

    // If the method failed to interpret the process arguments
    // show the command help
    if (!logger) {
        logger = commands.help(command);
        command = 'help';
    }
}

// Get the renderer and configure it with the executed command
renderer = getRenderer(command, logger.json, config);

logger
    .on('end', function (data) {
        if (!config.silent && !config.quiet) {
            renderer.end(data);
        }
    })
    .on('error', function (err)  {
        if (levels.error >= logLevel) {
            renderer.error(err);
        }

        process.exit(1);
    })
    .on('log', function (log) {
        if (levels[log.level] >= logLevel) {
            renderer.log(log);
        }
    })
    .on('prompt', function (prompt, callback) {
        renderer.prompt(prompt)
            .then(function (answer) {
                callback(answer);
            });
    });

// -- helpers
function expandNames(obj, prefix, stack) {
    prefix = prefix || '';
    stack = stack || [];

    mout.object.forOwn(obj, function (value, name) {
        name = prefix + name;

        stack.push(name);

        if (typeof value === 'object' && !value.line) {
            expandNames(value, name + ' ', stack);
        }
    });
    return stack;
}
function getRenderer(command, json, config) {
    if (config.json || json) {
        return new renderers.Json(command, config);
    }
    return new renderers.Standard(command, config);
}
