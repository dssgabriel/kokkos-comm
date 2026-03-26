// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Adapted from the OSU Benchmarks
// Copyright (c) 2002-2024 the Network-Based Computing Laboratory
// (NBCL), The Ohio State University.

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "KokkosComm/concepts.hpp"
#include "test_utils.hpp"

template <KokkosComm::KokkosView View>
void osu_latency_KC_core_send_recv(benchmark::State&, MPI_Comm, KokkosComm::Communicator<>& comm, const View& v) {
  if (comm.rank() == 0) {
    KokkosComm::send(comm, v, 1).wait();
  } else if (comm.rank() == 1) {
    KokkosComm::recv(comm, v, 0).wait();
  }
}
void benchmark_osu_latency_KC_core_send_recv(benchmark::State& state) {
  auto exec = Kokkos::DefaultExecutionSpace{};
  auto comm = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec);
  if (comm.size() != 2) {
    state.SkipWithError("benchmark_osu_latency_KC_core_send_recv needs exactly 2 ranks");
  }
  using View_t = Kokkos::View<char*>;
  View_t v("v", state.range(0));
  for (auto _ : state) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_KC_core_send_recv<View_t>, comm, v);
  }
  state.counters["bytes"] = v.size() * 2;
}
BENCHMARK(benchmark_osu_latency_KC_core_send_recv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);

template <KokkosComm::KokkosView View>
void osu_latency_KC_mpi_isend_irecv(benchmark::State&, MPI_Comm, KokkosComm::Communicator<>& comm, const View& v) {
  if (comm.rank() == 0) {
    KokkosComm::mpi::isend(comm, v, 1, 0).wait();
  } else if (comm.rank() == 1) {
    KokkosComm::mpi::irecv(comm, v, 0, 0).wait();
  }
}
void benchmark_osu_latency_KC_mpi_isend_irecv(benchmark::State& state) {
  auto exec = Kokkos::DefaultExecutionSpace{};
  auto comm = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec);
  if (comm.size() != 2) {
    state.SkipWithError("benchmark_osu_latency_KC_mpi_isend_irecv needs exactly 2 ranks");
  }
  using View_t = Kokkos::View<char*>;
  View_t v("v", state.range(0));
  for (auto _ : state) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_KC_mpi_isend_irecv<View_t>, comm, v);
  }
  state.counters["bytes"] = v.size() * 2;
}
BENCHMARK(benchmark_osu_latency_KC_mpi_isend_irecv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);

template <KokkosComm::KokkosView View>
void osu_latency_KC_mpi_send_recv(benchmark::State&, MPI_Comm, KokkosComm::Communicator<>& comm, const View& v) {
  if (comm.rank() == 0) {
    KokkosComm::mpi::send(comm, v, 1, 0);
  } else if (comm.rank() == 1) {
    KokkosComm::mpi::recv(comm, v, 0, 0);
  }
}
void benchmark_osu_latency_KC_mpi_send_recv(benchmark::State& state) {
  auto exec = Kokkos::DefaultExecutionSpace{};
  auto comm = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec);
  if (comm.size() != 2) {
    state.SkipWithError("benchmark_osu_latency_KC_mpi_send_recv needs exactly 2 ranks");
  }
  using View_t = Kokkos::View<char*>;
  View_t v("v", state.range(0));
  for (auto _ : state) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_KC_mpi_send_recv<View_t>, comm, v);
  }
  state.counters["bytes"] = v.size() * 2;
}
BENCHMARK(benchmark_osu_latency_KC_mpi_send_recv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);

template <KokkosComm::KokkosView View>
void osu_latency_MPI_isend_irecv(benchmark::State&, MPI_Comm comm, const View& v, int rank) {
  if (rank == 0) {
    MPI_Request rreq;
    MPI_Irecv(v.data(), v.size(), KokkosComm::datatype_for<KokkosComm::MpiSpace>(v), 1, 0, comm, &rreq);
    MPI_Wait(&rreq, MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Request sreq;
    MPI_Isend(v.data(), v.size(), KokkosComm::datatype_for<KokkosComm::MpiSpace>(v), 0, 0, comm, &sreq);
    MPI_Wait(&sreq, MPI_STATUS_IGNORE);
  }
}
void benchmark_osu_latency_MPI_isend_irecv(benchmark::State& state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    state.SkipWithError("benchmark_osu_latency_MPI_isend_irecv needs exactly 2 ranks");
  }
  using View_t = Kokkos::View<char*>;
  View_t v("v", state.range(0));
  for (auto _ : state) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_MPI_isend_irecv<View_t>, v, rank);
  }
  state.counters["bytes"] = v.size() * 2;
}
BENCHMARK(benchmark_osu_latency_MPI_isend_irecv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);

template <typename View>
void osu_latency_MPI_send_recv(benchmark::State&, MPI_Comm comm, const View& v, int rank) {
  if (rank == 0) {
    MPI_Recv(v.data(), v.size(), KokkosComm::datatype_for<KokkosComm::MpiSpace>(v), 1, 0, comm, MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Send(v.data(), v.size(), KokkosComm::datatype_for<KokkosComm::MpiSpace>(v), 0, 0, comm);
  }
}
void benchmark_osu_latency_MPI_send_recv(benchmark::State& state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    state.SkipWithError("benchmark_osu_latency_MPI_send_recv needs exactly 2 ranks");
  }
  using View_t = Kokkos::View<char*>;
  View_t v("v", state.range(0));
  for (auto _ : state) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_MPI_send_recv<View_t>, v, rank);
  }
  state.counters["bytes"] = v.size() * 2;
}
BENCHMARK(benchmark_osu_latency_MPI_send_recv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);
