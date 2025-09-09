// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iosfwd>
#include <type_traits>

#include <gtest/gtest.h>
#include <nccl.h>

#include <KokkosComm/KokkosComm.hpp>

#define NCCL_CHECK(cmd)                                                    \
  if (ncclResult_t res = cmd; res != ncclSuccess) {                        \
    std::cerr << std::format("NCCL error: {}\n", ncclGetErrorString(res)); \
    std::exit(EXIT_FAILURE);                                               \
  }

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Nccl;

template <typename T>
class P2P : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<float, double, int, unsigned, int64_t, size_t>;

TYPED_TEST_SUITE(P2P, ScalarTypes);

auto init_mpi() -> std::tuple<int, int> {
  static bool is_initialized = false;
  if (not is_initialized) {
    MPI_Init(nullptr, nullptr);
    is_initialized = true;
  }
  int np, me;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  return std::make_tuple(np, me);
}

auto create_nccl_comm() -> ncclComm_t {
  auto [np, me] = init_mpi();
  ncclUniqueId id;
  if (0 == me) {
    ncclGetUniqueId(&id);
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  ncclComm_t comm;
  ncclGroupStart();
  // Assume one device per process
  ncclCommInitRank(&comm, np, id, me);
  ncclGroupEnd();
  MPI_Barrier(MPI_COMM_WORLD);

  // We can finalize MPI since we only need it for initializing NCCL
  MPI_Finalize();
  return comm;
}

template <typename Scalar>
auto p2p_1d_contig() -> void {
  Kokkos::View<Scalar *> a("a", 1000);

  KokkosComm::Handle<ExecSpace, CommSpace> h(create_nccl_comm());
  if (h.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << h.size() << " provided)";
  }

  if (0 == h.rank()) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    auto req = KokkosComm::send(h, a, dst);
    KokkosComm::wait(req);
  } else if (1 == h.rank()) {
    int src  = 0;
    auto req = KokkosComm::recv(h, a, src);
    KokkosComm::wait(req);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != Scalar(i); }, errs);
    ASSERT_EQ(errs, 0);
  }
}

template <typename Scalar>
auto p2p_1d_noncontig() -> void {
  Kokkos::View<Scalar **, Kokkos::LayoutRight> b("a", 10, 10);
  auto a = Kokkos::subview(b, Kokkos::ALL, 2);  // take column 2 (non-contiguous)

  KokkosComm::Handle<ExecSpace, CommSpace> h(create_nccl_comm());
  if (h.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << h.size() << " provided)";
  }

  if (0 == h.rank()) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    KokkosComm::Req req = KokkosComm::send(h, a, dst);
    KokkosComm::wait(req);
  } else if (1 == h.rank()) {
    int src  = 0;
    auto req = KokkosComm::recv(h, a, src);
    KokkosComm::wait(req);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != Scalar(i); }, errs);
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(PointToPoint, Contiguous1D) { p2p_1d_contig<typename TestFixture::Scalar>(); }

TYPED_TEST(PointToPoint, NonContiguous1D) { p2p_1d_noncontig<typename TestFixture::Scalar>(); }

}  // namespace
