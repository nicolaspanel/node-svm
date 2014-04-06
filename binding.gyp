{
  'targets': [
    {
      'target_name': 'addon',
      'sources': [
        './src/libsvm-318/svm.cpp',
        './src/addon.cc',
        './src/node-svm/node-svm.cc'
      ],
      'cflags': ['-Wall', '-O3', '-fPIC', '-c'],
      "cflags_cc!": ["-fno-rtti", "-fno-exceptions"]
    }
  ]
}