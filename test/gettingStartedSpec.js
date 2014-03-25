var assert = require('assert'), 
    should = require('should'),
    addon = require('../build/Release/addon');
    
describe('simple hello world test', function(){
    it('should return world', function(){
        addon.hello().should.equal('World!');
    });
});