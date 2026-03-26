// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <mpi.h>
#include <KokkosComm/KokkosComm.hpp>

#include "KokkosComm/concepts.hpp"
#include "test_utils.hpp"

template <KokkosComm::KokkosView View>
void channel_send_recv(benchmark::State&, MPI_Comm, KokkosComm::Communicator<>& comm, const View& v, int rank) {
  const int size     = comm.size();
  const int dst_rank = (rank + 1) % size;         // send to next rank
  const int src_rank = (rank - 1 + size) % size;  // recv from prev rank
  KokkosComm::Channel<> channel(dst_rank, src_rank, 42, comm.comm());
  channel.sendinit(v);
  channel.recvinit(v);
  channel.start();
  channel.wait();
}

void benchmark_channel_send_recv(benchmark::State& state) {
  auto exec = Kokkos::DefaultExecutionSpace{};
  auto comm = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec);
  if (comm.size() < 2) {
    state.SkipWithError("benchmark_channel_send_recv needs at least 2 ranks");
  }

  using Scalar = double;
  using View_t = Kokkos::View<Scalar*>;
  View_t v("v", 1'000'000);

  for (auto _ : state) {
    do_iteration(state, MPI_COMM_WORLD, channel_send_recv<View_t>, comm, v, comm.rank());
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * v.size() * 2);
}

BENCHMARK(benchmark_channel_send_recv)->UseManualTime()->Unit(benchmark::kMillisecond);
