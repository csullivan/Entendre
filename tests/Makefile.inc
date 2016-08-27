EXE_SRC_FILES = run_all.cc
EXE_INCLUDE_DIRS = googletest/googletest/include googletest/googletest
EXTRA_SRC_FILES = $(patsubst $(CURDIR)/%,%,$(filter-out $(EXE_SRC_FILES),$(wildcard $(CURDIR)/*.cc))) googletest/googletest/src/gtest-all.cc

ALL_CPPFLAGS += -pthread
ALL_LDLIBS += -pthread
