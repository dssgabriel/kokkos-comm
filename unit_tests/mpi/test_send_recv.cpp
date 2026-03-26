// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "../view_utils.hpp"

namespace {

template <KokkosComm::KokkosView View, KokkosComm::mpi::CommunicationMode SendMode>
auto test_mpi_send_comm_mode_recv(const View& v, SendMode) -> void {
  if constexpr (std::is_same_v<SendMode, KokkosComm::mpi::CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  auto exec      = Kokkos::DefaultExecutionSpace{};
  auto comm      = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec);
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int src = 0;
  const int dst = 1;
  const int tag = 3;

  if (rank == src) {
    test_utils::init_view(exec, v);
    exec.fence();

    KokkosComm::mpi::send(comm, v, dst, tag, SendMode{});
  } else if (rank == dst) {
    KokkosComm::mpi::recv(comm, v, src, tag);

    int errs = test_utils::count_errors(v);
    ASSERT_EQ(errs, 0);
  }
}

template <typename T>
class MpiSendRecv : public ::testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(MpiSendRecv, ScalarTypes);

TYPED_TEST(MpiSendRecv, Contig1DStandard) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "v", 1013);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeStandard{});
}
TYPED_TEST(MpiSendRecv, Contig1DReady) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "v", 1013);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeReady{});
}
TYPED_TEST(MpiSendRecv, Contig1DSynchronous) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "v", 1013);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeSynchronous{});
}

TYPED_TEST(MpiSendRecv, NonContig1DStandard) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "v", 1013);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeStandard{});
}
TYPED_TEST(MpiSendRecv, NonContig1DReady) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "v", 1013);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeReady{});
}
TYPED_TEST(MpiSendRecv, NonContig1DSynchronous) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "v", 1013);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeSynchronous{});
}

TYPED_TEST(MpiSendRecv, Contig2DStandard) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "v", 137, 17);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeStandard{});
}
TYPED_TEST(MpiSendRecv, Contig2DReady) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "v", 137, 17);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeReady{});
}
TYPED_TEST(MpiSendRecv, Contig2DSynchronous) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "v", 137, 17);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeSynchronous{});
}

TYPED_TEST(MpiSendRecv, NonContig2DStandard) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "v", 137, 17);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeStandard{});
}
TYPED_TEST(MpiSendRecv, NonContig2DReady) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "v", 137, 17);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeReady{});
}
TYPED_TEST(MpiSendRecv, NonContig2DSynchronous) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "v", 137, 17);
  test_mpi_send_comm_mode_recv(v, KokkosComm::mpi::CommModeSynchronous{});
}

}  // namespace
