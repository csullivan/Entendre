#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..



# Build doxygen
doxygen scripts/doxy.config



# Run clang static analysis
scons -c
scan-build -o docs/temp scons
# Move output directory to correct location
mv docs/temp/* docs/clang-static-analysis
rmdir docs/temp



# Compile with gcov
scons -c
scons CODE_COVERAGE=1
# Collect stats
lcov --base-directory . --directory . --zerocounters -q
bin/entendre_tests
lcov --base-directory . --directory . -c -o docs/temp.info
# Clear library dependencies
lcov --remove docs/temp.info "/usr*" -o docs/temp.info
lcov --remove docs/temp.info "build/.dependencies*" -o docs/temp.info
# Generate HTML
rm -rf docs/code-coverage
genhtml -o docs/code-coverage -t "Entendre code coverage" docs/temp.info
mv docs/temp.info docs/code-coverage
