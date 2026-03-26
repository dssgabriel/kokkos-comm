// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <mpi.h>
#include <KokkosComm/KokkosComm.hpp>

#include "test_utils.hpp"

template <KokkosComm::KokkosView View>
void send_recv(benchmark::State&, MPI_Comm, KokkosComm::Communicator<>& comm, const View& v, int rank) {
  if (0 == rank) {
    KokkosComm::mpi::send(comm, v, 1, 0);
    KokkosComm::mpi::recv(comm, v, 1, 0);
  } else if (1 == rank) {
    KokkosComm::mpi::recv(comm, v, 0, 0);
    KokkosComm::mpi::send(comm, v, 0, 0);
  }
}

void benchmark_send_recv(benchmark::State& state) {
  auto exec = Kokkos::DefaultExecutionSpace{};
  auto comm = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec);
  if (comm.size() < 2) {
    state.SkipWithError("benchmark_send_recv needs at least 2 ranks");
  }

  using Scalar = double;
  using View_t = Kokkos::View<Scalar*>;
  View_t v("v", 1'000'000);

  for (auto _ : state) {
    do_iteration(state, MPI_COMM_WORLD, send_recv<View_t>, comm, v, comm.rank());
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * v.size() * 2);
}

BENCHMARK(benchmark_send_recv)->UseManualTime()->Unit(benchmark::kMillisecond);
