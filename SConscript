Import('env')

env.SharedLibraryDir('libFeedForward')
env.SharedLibraryDir('libNeat')
env.PythonLibraryDir('pyneat.so', 'libpyneat', dependencies=['Neat'])
env.UnitTestDir('entendre_tests','tests', extra_inc_dir='include')
env.MainDir('.')
