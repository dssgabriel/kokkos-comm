cmake_minimum_required(VERSION 3.25)

project(
  KokkosCommPerfTests
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Performance tests for the Kokkos Comm experimental explicit communication interfaces"
)

enable_testing()

option(
  KokkosComm_ENABLE_LARGE_SCALE_PERFTESTS
  "Register multi-node power iteration benchmarks requiring 512-1152 GiB of aggregate matrix storage"
  OFF
)
set(KOKKOSCOMM_BENCHMARKS_ENABLE_LARGE_SCALE ${KokkosComm_BENCHMARKS_ENABLE_LARGE_SCALE})

# Treat the perf tests as a separate project
# If not included in the Kokkos Comm build, find Kokkos Comm to do a standalone build
if(NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()

# MPI also bootstraps the NCCL benchmark and synchronizes benchmark timings.
find_package(MPI REQUIRED)
find_package(KokkosKernels REQUIRED)

include(FetchContent)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG
    192ef10025eb2c4cdd392bc502f0c852196baa48 # v1.9.5
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(kc-test)

kc_add_perf_test(
  bench.core.von_misses
  CORE
  NUM_PES 4
  FILES bench_main.cpp bench_von_misses.cpp
  LIBRARIES Kokkos::kokkoskernels
  ARGUMENTS --benchmark_filter=N:4096/ --benchmark_min_time=0.01s
)
if(KokkosComm_ENABLE_LARGE_SCALE_PERFTESTS)
  target_compile_definitions(bench.core.von_misses PRIVATE KOKKOSCOMM_ENABLE_LARGE_SCALE_PERFTESTS)
endif()
set_tests_properties(bench.core.von_misses PROPERTIES FAIL_REGULAR_EXPRESSION "ERROR OCCURRED")

if(KokkosComm_ENABLE_MPI)
  kc_add_perf_test(bench.mpi.p2p MPI NUM_PES 2 FILES bench_main.cpp mpi/test_sendrecv.cpp LIBRARIES KokkosComm::KokkosComm)
  kc_add_perf_test(bench.mpi.channel_p2p MPI NUM_PES 2 FILES bench_main.cpp mpi/test_channel_sendrecv.cpp LIBRARIES KokkosComm::KokkosComm)
  kc_add_perf_test(bench.mpi.2d_halo MPI NUM_PES 2 FILES bench_main.cpp mpi/test_2d_halo.cpp LIBRARIES KokkosComm::KokkosComm)
  kc_add_perf_test(bench.mpi.osu_latency MPI NUM_PES 2 FILES bench_main.cpp mpi/test_osu_latency.cpp LIBRARIES KokkosComm::KokkosComm)
endif()
