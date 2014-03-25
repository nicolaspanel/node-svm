module.exports = function (grunt) {
    grunt.initConfig({
        gyp: {
            addon: {}
        },
        mochaTest: {
            files: [ 'test/*Spec.js']
        },
        jshint: {   
            all: [ 'gruntfile.js', 'lib/*.js', 'test/*.js'],
            options: {
                globals: {
                    it: true,
                    describe: true,
                    beforeEach: true,
                    dateFormat: true
                },
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
                strict: false,
                es5: true
            }
        }
    });
    grunt.registerTask('default', ['jshint:all', 'gyp:addon', 'mochaTest']);
    grunt.loadNpmTasks('grunt-contrib-jshint');
    grunt.loadNpmTasks('grunt-mocha-test'); 
    grunt.loadNpmTasks('grunt-node-gyp');   
};