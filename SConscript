Import('env')

env.Append(LIBPATH='#/lib')
env.Append(RPATH=[Literal('\\$$ORIGIN')])
env.Append(RPATH=[Literal('\\$$ORIGIN/../lib')])
env.Append(CPPPATH=[Dir('include')])

libFeedForward = env.SConscript('libFeedForward/SConscript', exports='env')
libNeat = env.SConscript('libNeat/SConscript', exports='env')
libPyneat = env.SConscript('libpyneat/SConscript', exports='env libNeat')

# This assumes that a monkey-patched version is available
# env.append_lib(libFeedForward, libNeat)

# This one doesn't work.  Links against wrong library path
# env.Append(LIBS=[libFeedForward, libNeat])

# This one does work, but is ugly
env.Append(LIBPATH=[libFeedForward[0].dir, libNeat[0].dir])
env.Append(LIBS=[libFeedForward[0].name, libNeat[0].name])

run_all = env.SConscript('tests/SConscript', exports='env')

# for cc_file in Glob('*.cc'):
#     env.Program('bin/{}'.format(cc_file.filename[:-3]), cc_file)

train_xor = env.Program('train_xor.cc')
neat_xor = env.Program('neat_xor.cc')

env.Install('#/lib',[libFeedForward, libNeat, libPyneat])
env.Install('#/bin',[run_all, train_xor, neat_xor])

Clean('#','#/lib')
Clean('#','#/bin')
