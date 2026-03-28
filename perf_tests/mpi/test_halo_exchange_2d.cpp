// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "test_utils.hpp"

void noop(benchmark::State, MPI_Comm) {}

template <KokkosComm::KokkosView View>
void halo_exchange_2d(
    benchmark::State&, MPI_Comm, KokkosComm::Communicator<>& comm, const View& v, int nx, int ny, int rx, int ry, int rs
) {
  // 2D index of nbrs in minus and plus direction (periodic)
  const int xm1 = (rx + rs - 1) % rs;
  const int ym1 = (ry + rs - 1) % rs;
  const int xp1 = (rx + 1) % rs;
  const int yp1 = (ry + 1) % rs;

  // convert 2D rank into 1D rank
  auto get_rank = [=](const int x, const int y) -> int { return y * rs + x; };

  // send/recv subviews
  auto xp1_s = Kokkos::subview(v, v.extent(0) - 2, Kokkos::pair{1, ny + 1}, Kokkos::ALL);
  auto xp1_r = Kokkos::subview(v, v.extent(0) - 1, Kokkos::pair{1, ny + 1}, Kokkos::ALL);
  auto xm1_s = Kokkos::subview(v, 1, Kokkos::pair{1, ny + 1}, Kokkos::ALL);
  auto xm1_r = Kokkos::subview(v, 0, Kokkos::pair{1, ny + 1}, Kokkos::ALL);
  auto yp1_s = Kokkos::subview(v, Kokkos::pair{1, nx + 1}, v.extent(1) - 2, Kokkos::ALL);
  auto yp1_r = Kokkos::subview(v, Kokkos::pair{1, nx + 1}, v.extent(1) - 1, Kokkos::ALL);
  auto ym1_s = Kokkos::subview(v, Kokkos::pair{1, nx + 1}, 1, Kokkos::ALL);
  auto ym1_r = Kokkos::subview(v, Kokkos::pair{1, nx + 1}, 0, Kokkos::ALL);

  std::vector<KokkosComm::Request<>> reqs;
  // std::cerr << get_rank(rx, ry) << " -> " << get_rank(xp1, ry) << "\n";
  reqs.push_back(KokkosComm::sendrecv(comm, xp1_s, xp1_r, get_rank(xp1, ry)));
  reqs.push_back(KokkosComm::sendrecv(comm, xm1_s, xm1_r, get_rank(xm1, ry)));
  reqs.push_back(KokkosComm::sendrecv(comm, yp1_s, yp1_r, get_rank(rx, yp1)));
  reqs.push_back(KokkosComm::sendrecv(comm, ym1_s, ym1_r, get_rank(rx, ym1)));

  // wait for comm
  KokkosComm::wait_all(reqs);
}

void benchmark_halo_exchange_2d(benchmark::State& state) {
  using Scalar = double;
  using Grid_t = Kokkos::View<Scalar***, Kokkos::LayoutRight>;

  // problem size per rank
  int nx     = 512;
  int ny     = 512;
  int nprops = 3;

  auto exec = Kokkos::DefaultExecutionSpace{};
  auto comm = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec);

  const int size = comm.size();
  const int rank = comm.rank();

  const int rs = std::sqrt(size);
  const int rx = rank % rs;
  const int ry = rank / rs;

  if (rank < rs * rs) {
    // grid of elements, each with 3 properties, and a radius-1 halo
    Grid_t grid("grid", nx + 2, ny + 2, nprops);

    for (auto _ : state) {
      do_iteration(state, MPI_COMM_WORLD, halo_exchange_2d<Grid_t>, comm, grid, nx, ny, rx, ry, rs);
    }
  } else {
    for (auto _ : state) {
      do_iteration(state, MPI_COMM_WORLD, noop);  // do nothing...
    }
  }

  state.counters["active_ranks"] = rs * rs;
  state.counters["nx"]           = nx;
  // clang-format off
  state.SetBytesProcessed(
      sizeof(Scalar)
    * rs * rs // active ranks
    * state.iterations()
    * nprops
    * (
        2 * nx // send x nbrs
      + 2 * nx // recv x nbs
      + 2 * ny // send y nbrs
      + 2 * ny // recv y nbs
    )
  );
  // clang-format on
}

BENCHMARK(benchmark_halo_exchange_2d)->UseManualTime()->Unit(benchmark::kMillisecond);
