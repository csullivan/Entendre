Import('env')

env.OptionalCUDA()

env.Append(CPPPATH=['include'])
env.Append(CPPPATH=['glog/include'])
env.SharedLibraryDir('glog')

env.SharedLibraryDir('libFeedForward')
env.SharedLibraryDir('libEntendre', dependencies=['glog'])
env.SharedLibraryDir('libNeat', dependencies=['Entendre'])
env.PythonLibraryDir('pyneat.so', 'libpyneat', dependencies=['Neat','Entendre'])
env.UnitTestDir('entendre_tests','tests', extra_inc_dir='include')
env.MainDir('.')
