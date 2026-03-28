// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "../view_utils.hpp"

namespace {

template <KokkosComm::KokkosView SendV, KokkosComm::KokkosView RecvV>
auto test_mpi_sendrecv(const SendV& sv, const RecvV& rv) -> void {
  auto exec      = Kokkos::DefaultExecutionSpace{};
  auto comm      = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec);
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int peer = (rank + 1) % size;
  const int tag  = 7;

  test_utils::init_view(exec, sv);
  exec.fence();

  KokkosComm::mpi::isendrecv(comm, sv, peer, tag, rv, peer, tag).wait();

  int errs = test_utils::count_errors(rv);
  ASSERT_EQ(errs, 0);
}

template <typename T>
class MpiSendrecv : public ::testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(MpiSendrecv, ScalarTypes);

TYPED_TEST(MpiSendrecv, Contig1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "sv", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "rv", 1013);
  test_mpi_sendrecv(sv, rv);
}
TYPED_TEST(MpiSendrecv, NonContig1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "sv", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "rv", 1013);
  test_mpi_sendrecv(sv, rv);
}
TYPED_TEST(MpiSendrecv, Contig2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "sv", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "rv", 137, 17);
  test_mpi_sendrecv(sv, rv);
}
TYPED_TEST(MpiSendrecv, NonContig2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "sv", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "rv", 137, 17);
  test_mpi_sendrecv(sv, rv);
}

}  // namespace
