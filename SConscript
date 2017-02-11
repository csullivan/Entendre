Import('env')

env.SharedLibraryDir('libFeedForward')
env.SharedLibraryDir('libEntendre')
env.SharedLibraryDir('libNeat', dependencies=['Entendre'])
env.PythonLibraryDir('pyneat.so', 'libpyneat', dependencies=['Neat','Entendre'])
env.UnitTestDir('entendre_tests','tests', extra_inc_dir='include')
env.MainDir('.')
