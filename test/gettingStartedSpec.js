var assert = require('assert'), 
    should = require('should'),
    addon = require('../build/Release/addon');
    
describe('simple hello world test', function(){
    it('shouldbe able to return world synchronously', function(){
        addon.hello().should.equal('World!');
    });

    it('shouldbe able to return world asynchronously', function(done){
        addon.helloAsync(function(msg){
          msg.should.equal('World!');
          done();
        });
    });
});