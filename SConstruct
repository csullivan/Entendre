import os

import SCons

env = Environment(ENV = os.environ)
env.Append(CCFLAGS=['-std=c++1y','-pthread','-O3'])
env.Append(LINKFLAGS=['-pthread'])

def append_lib(self, *libs):
    for lib in libs:
        if isinstance(lib, str):
            libs = File(lib)

        if isinstance(lib, SCons.Node.FS.File):
            self.Append(LIBPATH=[lib.dir])
            self.Append(LIBS=[lib.name])
        else:
            self.append_lib(*lib)
type(env).append_lib = append_lib

# More readable output
if not ARGUMENTS.get('VERBOSE'):
    env['CXXCOMSTR'] = 'Compiling C++ object $TARGETS'
    env['CCCOMSTR'] = 'Compiling C object $TARGETS'
    env['ARCOMSTR'] = 'Packing static library $TARGETS'
    env['RANLIBCOMSTR'] = 'Indexing static library $TARGETS'
    env['SHCCCOMSTR'] = 'Compiling shared C object $TARGETS'
    env['SHCXXCOMSTR'] = 'Compiling shared C++ object $TARGETS'
    env['LINKCOMSTR'] = 'Linking $TARGETS'
    env['SHLINKCOMSTR'] = 'Linking shared $TARGETS'

env.SConscript('SConscript', exports='env', duplicate=True,
               variant_dir='build')
Clean('.','build')
