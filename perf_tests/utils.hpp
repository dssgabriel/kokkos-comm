// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <chrono>

#include <benchmark/benchmark.h>
#include <mpi.h>

#include "../unit_tests/logging.hpp"

template <typename F, typename... Args>
auto do_iteration(benchmark::State& state, F&& func, Args&&... args) -> void {
  using Clock    = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;

  auto start = Clock::now();
  func(state, std::forward<Args>(args)...);
  Duration elapsed = Clock::now() - start;

  double loc_lat = elapsed.count();
  double locsq   = loc_lat * loc_lat;

  double min_lat, max_lat, sum, sumsq;
  MPI_Allreduce(&loc_lat, &min_lat, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_lat, &max_lat, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_lat, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&locsq, &sumsq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  double avg_lat = sum / static_cast<double>(nranks);
  double stddev  = std::sqrt(std::max(0.0, (sumsq / nranks) - (avg_lat * avg_lat)));
  double cv      = (avg_lat > 0.0) ? (stddev / avg_lat) : 0.0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (cv >= 0.05 and rank == 0) {
    // KC_WARN("high variance between ranks ({}%)", cv * 100.0);
  }

  // Primary benchmark time
  state.SetIterationTime(max_lat);

  // Supplemental counters
  state.counters["min_rank_lat_ns"] = min_lat;
  state.counters["avg_rank_lat_ns"] = avg_lat;
  state.counters["stddev_lat_ns"]   = stddev;
}
