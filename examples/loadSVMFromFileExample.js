/**
  Simple example using C-SVC classificator to predict the xor function
  Dataset : no dataset used because we restore an existing 
            svm previously saved on a file
  
  Note : To save an existing svm, make sure it is trained and use svm#saveToFile(path).
**/
'use strict';

var libsvm = require('../lib/nodesvm');

var svm = new libsvm.SVM({
  file: './examples/models/xor.model'
});

console.log("xor trainned");
[0,1].forEach(function(a){
  [0,1].forEach(function(b){
    console.log("%d XOR %d => %d", a, b, svm.predict([a, b]));
  });
});

/* result : 
0 XOR 0 => 0
0 XOR 1 => 1
1 XOR 0 => 1
1 XOR 1 => 0
*/