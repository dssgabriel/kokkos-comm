// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

namespace {

using namespace KokkosComm::mpi;

template <typename T>
class MpiSendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(MpiSendRecv, ScalarTypes);

template <CommunicationMode SendMode, KokkosComm::KokkosView View1D>
void send_1d_comm_mode(const View1D &v) {
  if constexpr (std::is_same_v<SendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  static_assert(View1D::rank == 1, "");
  using Scalar = typename View1D::non_const_value_type;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  using Ex = Kokkos::DefaultExecutionSpace;
  Ex exec();

  if (0 == rank) {
    const int dst = 1;
    using Pol     = Kokkos::RangePolicy<Ex>;
    Pol policy(exec, 0, v.extent(0));

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const int i) { v(i) = i; }
    );
    exec.fence();

    KokkosComm::mpi::send(exec, v, dst, 0, MPI_COMM_WORLD, SendMode{});
  } else if (1 == rank) {
    const int src = 0;

    KokkosComm::mpi::recv(exec, v, src, 0, MPI_COMM_WORLD);

    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += v(i) != i; }, errs
    );
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(MpiSendRecv, 1D_contig_standard) {
  auto v = ViewBuilder<typename TestFixture::Scalar, 1>::view(Contig{}, "v", 1013);
  send_1d_comm_mode<CommModeStandard, decltype(v)>(v);
}
TYPED_TEST(MpiSendRecv, 1D_contig_ready) {
  auto v = ViewBuilder<typename TestFixture::Scalar, 1>::view(Contig{}, "v", 1013);
  send_1d_comm_mode<CommModeReady, decltype(v)>(v);
}
TYPED_TEST(MpiSendRecv, 1D_contig_synchronous) {
  auto v = ViewBuilder<typename TestFixture::Scalar, 1>::view(Contig{}, "v", 1013);
  send_1d_comm_mode<CommModeSynchronous, decltype(v)>(v);
}

TYPED_TEST(MpiSendRecv, 1D_noncontig_standard) {
  auto v = ViewBuilder<typename TestFixture::Scalar, 1>::view(NonContig{}, "v", 1013);
  send_1d_comm_mode<CommModeStandard, typename TestFixture::Scalar>();
}
TYPED_TEST(MpiSendRecv, 1D_noncontig_ready) {
  auto v = ViewBuilder<typename TestFixture::Scalar, 1>::view(NonContig{}, "v", 1013);
  send_1d_comm_mode<CommModeReady, typename TestFixture::Scalar>();
}
TYPED_TEST(MpiSendRecv, 1D_noncontig_synchronous) {
  auto v = ViewBuilder<typename TestFixture::Scalar, 1>::view(NonContig{}, "v", 1013);
  send_1d_comm_mode<CommModeSynchronous, typename TestFixture::Scalar>();
}

}  // namespace
