#!/bin/bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..



# Build doxygen
doxygen scripts/doxy.config



# Run clang static analysis
scons -c
scan-build-4.0 --use-c++=clang++-4.0 -o docs/temp scons
# Move output directory to correct location
mv docs/temp/* docs/clang-static-analysis
rmdir docs/temp



# Compile with gcov
scons -c
CXX=g++-6 scons CODE_COVERAGE=1 VERBOSE=1
lcov --version
# Collect stats
lcov --gcov-tool gcov-6 --base-directory . --directory . --zerocounters -q
bin/entendre_tests
lcov --gcov-tool gcov-6 --base-directory . --directory . -c -o docs/temp.info
# Clear library dependencies
lcov --remove docs/temp.info "/usr/*" -o docs/temp.info
lcov --remove docs/temp.info "*/build/.dependencies/*" -o docs/temp.info
# Generate HTML
rm -rf docs/code-coverage
genhtml -o docs/code-coverage -t "Entendre code coverage" docs/temp.info
mv docs/temp.info docs/code-coverage/data.info
