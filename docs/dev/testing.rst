*******
Testing
*******

The unit tests (``unit_tests``) and performance tests/benchmarks (``perf_tests``) are conceptually maintained as separate CMake projects contained in the same source tree.
This enforces some discipline with respect to testing the KokkosComm installation, since the tests don't have any special privileges.
During a build with tests enabled, they just get included at the very end of the build.


Testing the install
====================

1. Do a standard KokkosComm build and install with tests disabled (see the `Setup section <../getting_started/setup.html>`_). This build does not need to include unit tests or performance tests -- they will later be compiled against this install.

2. Do another build with the ``unit_tests`` directory as the source. You will need to provide ``-DKokkosComm_ROOT=<KOKKOSCOMM_INSTALL_DIR>/lib64/cmake`` just as any other external project would.

3. Run the tests on that build.

4. Repeat step 2 but with ``perf_tests`` as the source directory.

5. Run the tests on that build.


Complete workflow
-----------------

.. code-block:: console

    $ set -eou pipefail

    $ # Clone KokkosComm
    $ git clone https://github.com/kokkos/kokkos-comm.git
    $ cd kokkos-comm

    $ # Set the required path variables
    $ export KOKKOS_SRC_DIR=$PWD/kokkos
    $ export KOKKOS_BUILD_DIR=$PWD/kokkos-build
    $ export KOKKOS_INSTALL_DIR=$PWD/kokkos-install
    $ export KOKKOSCOMM_SRC_DIR=$PWD
    $ export KOKKOSCOMM_BUILD_DIR=build
    $ export KOKKOSCOMM_INSTALL_DIR=$PWD/install
    $ export KOKKOSCOMM_UNIT_TESTS_BUILD_DIR=build-unit-tests
    $ export KOKKOSCOMM_PERF_TESTS_BUILD_DIR=build-perf-tests

    $ # Clone Kokkos
    $ git clone https://github.com/kokkos/kokkos.git --branch master --depth 1 "$KOKKOS_SRC_DIR" || true

    $ # Configure Kokkos
    $ cmake -S "$KOKKOS_SRC_DIR" \
            -B "$KOKKOS_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKokkos_ENABLE_SERIAL=ON \
            -DKokkos_ENABLE_OPENMP=ON

    $ # Build Kokkos
    $ cmake --build "$KOKKOS_BUILD_DIR" --parallel $(nproc)

    $ # Install Kokkos
    $ cmake --install "$KOKKOS_BUILD_DIR" --prefix "$KOKKOS_INSTALL_DIR"

    $ # Configure KokkosComm
    $ [ -d "$KOKKOSCOMM_BUILD_DIR" ] && rm --recursive --force "$KOKKOSCOMM_BUILD_DIR"
    $ cmake -S "$KOKKOSCOMM_SRC_DIR" \
            -B "$KOKKOSCOMM_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKokkos_ROOT="$KOKKOS_INSTALL_DIR" \
            -DKokkosComm_ENABLE_TESTS=OFF \
            -DKokkosComm_ENABLE_PERFTESTS=OFF

    $ # Build KokkosComm
    $ cmake --build "$KOKKOSCOMM_BUILD_DIR" --parallel $(nproc) --verbose

    $ # Install KokkosComm
    $ [ -d "$KOKKOSCOMM_INSTALL_DIR" ] && rm --recursive --force "$KOKKOSCOMM_INSTALL_DIR"
    $ cmake --install "$KOKKOSCOMM_BUILD_DIR" --prefix "$KOKKOSCOMM_INSTALL_DIR"

    $ # Remove KokkosComm build files
    $ rm --recursive --force "$KOKKOSCOMM_BUILD_DIR"

    $ # Configure KokkosComm unit tests
    $ rm --recursive --force "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR"
    $ cmake -S "$KOKKOSCOMM_SRC_DIR"/unit_tests \
            -B "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKokkos_ROOT="$KOKKOS_INSTALL_DIR" \
            -DKokkosComm_ROOT="$KOKKOSCOMM_INSTALL_DIR"

    $ # Build KokkosComm unit tests
    $ cmake --build "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR" --parallel $(nproc) --verbose

    $ # Run KokkosComm unit tests
    $ ctest -V --test-dir "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR"

    $ # Configure KokkosComm performance tests
    $ rm --recursive --force "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR"
    $ cmake -S "$KOKKOSCOMM_SRC_DIR"/perf_tests \
            -B "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKokkos_ROOT="$KOKKOS_INSTALL_DIR" \
            -DKokkosComm_ROOT="$KOKKOSCOMM_INSTALL_DIR"

    $ # Build KokkosComm performance tests
    $ cmake --build "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR" --parallel $(nproc) --verbose

    $ # Run KokkosComm performance tests
    $ ctest -V --test-dir "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR"
