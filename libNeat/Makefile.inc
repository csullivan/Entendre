# This file can be placed inside of a library directory to customize
# the behavior of that library. Each option, if left commented, will
# assumed its default value.

# The name of the library.
# Defaults to LIBNAME, where libLIBNAME is the directory.
LIBRARY_NAME = $(patsubst lib%,%,$(notdir $(CURDIR)))

# The flag that will be passed to the include the library in
# executables.
LIBRARY_FLAG = -l$(LIBRARY_NAME)

# The directories containing source files for the library.
LIBRARY_SRC_DIRS = src

# The directories containing include files for the library. These
# directories will be added to the include path for all files in the
# project.
LIBRARY_INCLUDE_DIRS = include

# The directories containing include files for the library.  These
# directories will be added to the include path only for files within
# this library
LIBRARY_PRIVATE_INCLUDE_DIRS = ../pybind11/include ../../../../../../../../../../usr/include/python3.5m

# A space-delimited list of file patterns to be excluded
# For example, this may contain a source file or directory
# that is not to be compiled in the current build.
# % can be used to create wildcard exclusions.
# Note: Files in the library's directory should be listed as "filename",
#    not "./filename".
LIBRARY_EXCLUSIONS =

# Extra flags that should be present when linking the shared library.
# This may include other libraries that should be included.
$(LIBRARY): SHARED_LDLIBS +=

# Compiler flag overrides for src files within this library.
$(LIBRARY):
