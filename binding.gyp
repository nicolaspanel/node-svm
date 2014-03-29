{
  "targets": [
    {
      "target_name": "addon",
      "sources": [
        "./src/addon.cc",
        "./src/node-svm/node-svm.cc",
        "./src/libsvm-317/svm.cpp"
      ],
      "cflags": ["-Wall"]
    }
  ]
}