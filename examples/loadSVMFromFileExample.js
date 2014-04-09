/**
  Simple example using C-SVC classificator to predict the xor function
  Dataset : no dataset used because we restore an existing 
            svm previously saved on a file
  
  Note : To save an existing svm, make sure it is trained and use svm#saveToFile(path).
**/
'use strict';

var nodesvm = require('../lib/nodesvm');

var svm = new nodesvm.loadSvmFromFile('./examples/models/xor.model');

//NODE : no need to retrain the svm
[0,1].forEach(function(a){
  [0,1].forEach(function(b){
    console.log("%d XOR %d => %d", a, b, svm.predict([a, b]));
  });
});

/* OUTPUT 
0 XOR 0 => 0
0 XOR 1 => 1
1 XOR 0 => 1
1 XOR 1 => 0
*/