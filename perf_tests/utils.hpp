// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <chrono>
#include <cstddef>
#include <utility>

#include <benchmark/benchmark.h>
#include <mpi.h>

template <typename F, typename... Args>
auto do_iteration(benchmark::State& state, F&& func, Args&&... args) -> void {
  using Clock    = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;

  auto start = Clock::now();
  func(state, std::forward<Args>(args)...);
  Duration elapsed = Clock::now() - start;

  double max_elapsed_second;
  double elapsed_seconds = elapsed.count();
  MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  state.SetIterationTime(max_elapsed_second);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    state.counters["rank0_ping_pong_latency"] = elapsed_seconds;
  }
}

namespace collective_bw {

inline constexpr int warmup_iterations = 1;
// nccl-tests uses 20 timed operations by default. We adopt that reference
// value on the assumption that it is a sound default for collective bandwidth
// measurements. See NVIDIA/nccl-tests src/common.cu (`iters = 20`):
// https://github.com/NVIDIA/nccl-tests/blob/master/src/common.cu
inline constexpr int timed_iterations = 20;

template <typename RunBatch>
auto run(benchmark::State& state, RunBatch&& run_batch, std::size_t algorithm_bytes, double bus_bw_factor) -> void {
  // Warm up lazy transport setup and collective algorithm selection before timing.
  run_batch(warmup_iterations);

  for (auto _ : state) {
    // Start all ranks from the same point. This barrier is intentionally not timed.
    MPI_Barrier(MPI_COMM_WORLD);

    const auto start = std::chrono::steady_clock::now();
    run_batch(timed_iterations);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    const double local_elapsed_seconds = elapsed.count();
    double max_elapsed_seconds         = 0.0;
    MPI_Allreduce(&local_elapsed_seconds, &max_elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    state.SetIterationTime(max_elapsed_seconds / timed_iterations);
  }

  using Counter           = benchmark::Counter;
  state.counters["bytes"] = static_cast<double>(algorithm_bytes);
  // Algorithm bandwidth measures useful application payload per second and is
  // the primary metric for comparing Kokkos Comm with its raw backend. Bus
  // bandwidth applies the nccl-tests collective-specific normalization so that
  // hardware-link utilization can be compared across communicator sizes.
  state.counters["algbw"] =
      Counter(static_cast<double>(algorithm_bytes), Counter::kIsIterationInvariantRate, Counter::OneK::kIs1000);
  state.counters["busbw"] = Counter(
      static_cast<double>(algorithm_bytes) * bus_bw_factor, Counter::kIsIterationInvariantRate, Counter::OneK::kIs1000
  );
}

inline auto require_multiple_ranks(benchmark::State& state, int size, const char* benchmark_name) -> bool {
  if (size >= 2) {
    return true;
  }
  state.SkipWithError(benchmark_name);
  return false;
}

inline auto allreduce_bus_bw_factor(int size) -> double {
  return 2.0 * static_cast<double>(size - 1) / static_cast<double>(size);
}

inline auto allgather_alltoall_bus_bw_factor(int size) -> double {
  return static_cast<double>(size - 1) / static_cast<double>(size);
}

}  // namespace collective_bw
