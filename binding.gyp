{
  'targets': [
    {
      'target_name': 'addon',
      'sources': [
        './src/libsvm/svm.cpp',
        './src/addon.cc',
        './src/node-svm/node-svm.cc'
      ],
      "include_dirs" : [
        "<!(node -e \"require('nan')\")"
      ],
      'cflags': ['-Wall', '-O3', '-fPIC', '-c'],
      "cflags_cc!": ["-fno-rtti", "-fno-exceptions"]
    }
  ]
}
