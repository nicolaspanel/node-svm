'use strict';

module.exports = function (grunt) {
    require('load-grunt-tasks')(grunt);
    grunt.initConfig({
        gyp: {
            addon: {}
        },
        simplemocha: {
            options: {
                reporter: 'spec',
                timeout: '5000'
            },
            full: {
                src: ['test/**/*.spec.js']
            },
            short: {
                options: {
                    reporter: 'dot'
                },
                src: ['test/*.spec.js']
            }
        },
        jshint: {
            options: {
                jshintrc: '.jshintrc'
            },
            files: [
                'Gruntfile.js',
                'bin/*',
                'lib/**/*.js',
                'test/**/*.js',
                'examples/**/*.js',
                '!test/reports/**/*'
            ]
        },
        exec: {
            coveralls: {
                command: 'STRICT_REQUIRE=1 node node_modules/.bin/istanbul cover ./node_modules/mocha/bin/_mocha --report lcovonly -- -R dot test/**/*.spec.js && cat ./coverage/lcov.info | ./node_modules/.bin/coveralls && rm -rf ./coverage'
            }
        },
        watch: {
            files: ['<%= jshint.files %>'],
            tasks: ['jshint', 'simplemocha:short']
        }
    });
    grunt.registerTask('test', ['jshint', 'simplemocha:full']);
    grunt.registerTask('travis', ['test', 'exec:coveralls']);

    grunt.registerTask('default', 'test');
};