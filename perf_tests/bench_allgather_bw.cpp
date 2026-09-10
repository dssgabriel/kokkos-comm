// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>
#include <mpi.h>
#ifdef KOKKOSCOMM_ENABLE_NCCL
#include <nccl.h>
#endif

#include "utils.hpp"
#ifdef KOKKOSCOMM_ENABLE_NCCL
#include "../unit_tests/nccl/utils.hpp"
#endif

namespace {

namespace KC  = KokkosComm;
namespace KCE = KC::Experimental;

using Scalar = float;
using DES    = Kokkos::DefaultExecutionSpace;
using View   = Kokkos::View<Scalar*>;

struct MessageShape {
  std::size_t contribution_count;
  std::size_t algorithm_bytes;
};

auto message_shape(benchmark::State& state, int size) -> MessageShape {
  const auto requested_bytes = static_cast<std::size_t>(state.range(0));
  const auto count           = requested_bytes / (static_cast<std::size_t>(size) * sizeof(Scalar));
  return {count, count * static_cast<std::size_t>(size) * sizeof(Scalar)};
}

auto initialize(const View& sv, const View& rv) -> void {
  Kokkos::deep_copy(sv, Scalar{1});
  Kokkos::deep_copy(rv, Scalar{0});
  Kokkos::fence("initialize allgather bandwidth buffers");
}

auto make_receive_buffers(std::size_t count) -> std::vector<View> {
  // Every outstanding MPI collective needs non-overlapping receive storage.
  std::vector<View> buffers;
  buffers.reserve(collective_bw::timed_iterations);
  for (int i = 0; i < collective_bw::timed_iterations; ++i) {
    buffers.emplace_back("allgather_rv", count);
    Kokkos::deep_copy(buffers.back(), Scalar{0});
  }
  Kokkos::fence("initialize allgather bandwidth receive buffers");
  return buffers;
}

auto bench_allgather_bw_KC_MpiSpace(benchmark::State& state) -> void {
  auto comm = KC::Communicator<KC::MpiSpace, DES>::from_raw(MPI_COMM_WORLD, DES{});
  if (!collective_bw::require_multiple_ranks(
          state, comm.size(), "KokkosComm::MpiSpace allgather bandwidth benchmark needs at least 2 ranks"
      )) {
    return;
  }

  const auto shape = message_shape(state, comm.size());
  View sv("allgather_sv", shape.contribution_count);
  Kokkos::deep_copy(sv, Scalar{1});
  auto receive_buffers = make_receive_buffers(shape.contribution_count * static_cast<std::size_t>(comm.size()));

  std::vector<KC::Request<KC::MpiSpace>> requests;
  requests.reserve(collective_bw::timed_iterations);

  const auto run_batch = [&](int iterations) {
    for (int i = 0; i < iterations; ++i) {
      requests.emplace_back(KCE::allgather(comm, sv, receive_buffers[i]));
    }
    KC::wait_all(requests);
    requests.clear();
  };
  collective_bw::run(
      state, run_batch, shape.algorithm_bytes, collective_bw::allgather_alltoall_bus_bw_factor(comm.size())
  );
}

auto bench_allgather_bw_MPI_nb(benchmark::State& state) -> void {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (!collective_bw::require_multiple_ranks(
          state, size, "MPI nonblocking allgather bandwidth benchmark needs at least 2 ranks"
      )) {
    return;
  }

  const auto shape = message_shape(state, size);
  View sv("allgather_sv", shape.contribution_count);
  Kokkos::deep_copy(sv, Scalar{1});
  auto receive_buffers = make_receive_buffers(shape.contribution_count * static_cast<std::size_t>(size));

  std::vector<MPI_Request> requests(collective_bw::timed_iterations, MPI_REQUEST_NULL);

  const auto run_batch = [&](int iterations) {
    for (int i = 0; i < iterations; ++i) {
      MPI_Iallgather(
          sv.data(), static_cast<int>(shape.contribution_count), MPI_FLOAT, receive_buffers[i].data(),
          static_cast<int>(shape.contribution_count), MPI_FLOAT, MPI_COMM_WORLD, &requests[i]
      );
    }
    MPI_Waitall(iterations, requests.data(), MPI_STATUSES_IGNORE);
  };
  collective_bw::run(state, run_batch, shape.algorithm_bytes, collective_bw::allgather_alltoall_bus_bw_factor(size));
}

#ifdef KOKKOSCOMM_ENABLE_NCCL
using CuES = Kokkos::Cuda;

auto bench_allgather_bw_KC_NcclSpace(benchmark::State& state) -> void {
  auto& nccl_ctx = test_utils::NcclCtx::get();
  auto comm      = KC::Communicator<KCE::NcclSpace, CuES>::from_raw(nccl_ctx.comm(), CuES{});
  if (!collective_bw::require_multiple_ranks(
          state, comm.size(), "KokkosComm::NcclSpace allgather bandwidth benchmark needs at least 2 ranks"
      )) {
    return;
  }

  const auto shape = message_shape(state, comm.size());
  View sv("allgather_sv", shape.contribution_count);
  View rv("allgather_rv", shape.contribution_count * static_cast<std::size_t>(comm.size()));
  initialize(sv, rv);

  std::vector<KC::Request<KCE::NcclSpace>> requests;
  requests.reserve(collective_bw::timed_iterations);

  const auto run_batch = [&](int iterations) {
    for (int i = 0; i < iterations; ++i) {
      requests.emplace_back(KCE::allgather(comm, sv, rv));
    }
    KC::wait_all(requests);
    requests.clear();
  };
  collective_bw::run(
      state, run_batch, shape.algorithm_bytes, collective_bw::allgather_alltoall_bus_bw_factor(comm.size())
  );
}

auto bench_allgather_bw_NCCL(benchmark::State& state) -> void {
  auto& nccl_ctx = test_utils::NcclCtx::get();
  const int size = nccl_ctx.size();
  if (!collective_bw::require_multiple_ranks(
          state, size, "NCCL allgather bandwidth benchmark needs at least 2 ranks"
      )) {
    return;
  }

  const auto shape = message_shape(state, size);
  View sv("allgather_sv", shape.contribution_count);
  View rv("allgather_rv", shape.contribution_count * static_cast<std::size_t>(size));
  initialize(sv, rv);

  const auto run_batch = [&](int iterations) {
    for (int i = 0; i < iterations; ++i) {
      ncclAllGather(sv.data(), rv.data(), shape.contribution_count, ncclFloat, nccl_ctx.comm(), nccl_ctx.stream());
    }
    cudaStreamSynchronize(nccl_ctx.stream());
  };
  collective_bw::run(state, run_batch, shape.algorithm_bytes, collective_bw::allgather_alltoall_bus_bw_factor(size));
}
#endif

constexpr auto min_message_bytes = 1 << 10;
constexpr auto max_message_bytes = 1 << 28;

BENCHMARK(bench_allgather_bw_KC_MpiSpace)
    ->ArgName("bytes")
    ->RangeMultiplier(8)
    ->Range(min_message_bytes, max_message_bytes)
    ->Iterations(1)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_allgather_bw_MPI_nb)
    ->ArgName("bytes")
    ->RangeMultiplier(8)
    ->Range(min_message_bytes, max_message_bytes)
    ->Iterations(1)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
#ifdef KOKKOSCOMM_ENABLE_NCCL
BENCHMARK(bench_allgather_bw_KC_NcclSpace)
    ->ArgName("bytes")
    ->RangeMultiplier(8)
    ->Range(min_message_bytes, max_message_bytes)
    ->Iterations(1)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_allgather_bw_NCCL)
    ->ArgName("bytes")
    ->RangeMultiplier(8)
    ->Range(min_message_bytes, max_message_bytes)
    ->Iterations(1)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
#endif

}  // namespace
