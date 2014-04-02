{
  'targets': [
    {
      'target_name': 'addon',
      'sources': [
        './src/libsvm-317/svm.cpp',
        './src/addon.cc',
        './src/node-svm/node-svm.cc'
      ],
      'cflags': ['-Wall', '-O3', '-fPIC', '-shared']
    }
  ]
}