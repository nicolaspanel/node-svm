'use strict';

module.exports = function (grunt) {
  grunt.initConfig({
    gyp: {
      addon: {}
    },
    mochacov: {
      coverage: {
        options: {
          instrument: true,
          reporter: 'mocha-lcov-reporter',
          coveralls: {
            serviceName: 'travis-ci',
            repoToken: 'XRzdgZmcxy0P8tWmIKtv4LssCLYZ2LrKy'
          }
        }
      },
      test: {
        options: {
          reporter: 'spec'
        }
      },
      options: {
        files: 'test/*Spec.js'
      }
    },
    jshint: {   
      all: [ 'gruntfile.js', 'lib/*.js', 'test/*.js', 'examples/*.js',],
      options: {
        globals: {
          it: true,
          describe: true,
          beforeEach: true,
          afterEach: true,
          $V: true,
          $M: true
        },
        browser: false,
        bitwise: true,
        indent: 2,
        curly: true,
        eqeqeq: true,
        immed: true,
        latedef: true,
        newcap: true,
        noarg: true,
        sub: true,
        undef: true,
        boss: true,
        eqnull: true,
        node: true,
        strict: true,
        expr: true,
        es5: true,
        camelcase: true,
        smarttabs: true
      }
    }
  });
  grunt.registerTask('default', ['jshint:all', 'gyp:addon', 'mochacov:test']);
  grunt.registerTask('cov', ['mochacov:coverage']);
  grunt.loadNpmTasks('grunt-contrib-jshint');
  grunt.loadNpmTasks('grunt-node-gyp');   
  grunt.loadNpmTasks('grunt-mocha-cov');   
};