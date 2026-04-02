// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#include "view_builder.hpp"
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

using Ex = Kokkos::DefaultExecutionSpace;
using Co = KokkosComm::DefaultCommunicationSpace;

template <typename T>
class SendRecv : public testing::Test {
 public:
  using Scalar = T;
};

#if defined(KOKKOSCOMM_ENABLE_NCCL)
using ScalarTypes = ::testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    ::testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(SendRecv, ScalarTypes);

template <KokkosComm::KokkosView View1D>
void test_1d(const View1D &v) {
  static_assert(View1D::rank == 1, "");
  using Scalar = typename View1D::non_const_value_type;

#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto exec      = Ex();
  auto comm      = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int src = 0;
  const int dst = 1;

  if (rank == src) {
    Kokkos::parallel_for(
        v.extent(0), KOKKOS_LAMBDA(const int i) { v(i) = i; }
    );
    KokkosComm::send(comm, v, dst).wait();
  } else if (rank == dst) {
    KokkosComm::recv(comm, v, src).wait();

    int errs;
    Kokkos::parallel_reduce(
        v.extent(0), KOKKOS_LAMBDA(const int i, int &lsum) { lsum += v(i) != Scalar(i); }, errs
    );
    ASSERT_EQ(errs, 0);
  }
}

template <KokkosComm::KokkosView View2D>
void test_2d(const View2D &v) {
  static_assert(View2D::rank == 2, "");
  using Scalar = typename View2D::non_const_value_type;

#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto exec      = Ex();
  auto comm      = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int src = 0;
  const int dst = 1;

  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Policy policy(exec, {0, 0}, {v.extent(0), v.extent(1)});

  if (rank == src) {
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const int i, const int j) { v(i, j) = i * v.extent(0) + j; }
    );
    exec.fence();

    KokkosComm::send(comm, v, dst).wait();
  } else if (rank == dst) {
    KokkosComm::recv(comm, v, src).wait();

    int errs;
    Kokkos::parallel_reduce(
        policy, KOKKOS_LAMBDA(const int i, const int j, int &lsum) { lsum += v(i, j) != Scalar(i * v.extent(0) + j); },
        errs
    );
    exec.fence();
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(SendRecv, 1D_contig) {
  auto v = ViewBuilder<typename TestFixture::Scalar, 1>::view(contig{}, "v", 1013);
  test_1d(v);
}
TYPED_TEST(SendRecv, 2D_contig) {
  auto v = ViewBuilder<typename TestFixture::Scalar, 2>::view(contig{}, "v", 137, 17);
  test_2d(v);
}

}  // namespace
