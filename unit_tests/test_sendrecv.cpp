// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#include "view_utils.hpp"
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

template <KokkosComm::KokkosView SendV, KokkosComm::KokkosView RecvV>
void test_core_sendrecv(const SendV& sv, const RecvV& rv) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto exec      = Kokkos::DefaultExecutionSpace{};
  auto comm      = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int peer = (rank + 1) % size;

  test_utils::init_view(exec, sv);
  exec.fence();

  KokkosComm::sendrecv(comm, sv, rv, peer).wait();

  int errs = test_utils::count_errors(rv);
  ASSERT_EQ(errs, 0);
}

template <typename T>
class SendRecv : public ::testing::Test {
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

TYPED_TEST(SendRecv, Contig1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "v", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "v", 1013);
  test_core_sendrecv(sv, rv);
}
TYPED_TEST(SendRecv, NonContig1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "v", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "v", 1013);
  test_core_sendrecv(sv, rv);
}
TYPED_TEST(SendRecv, Contig2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "v", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "v", 137, 17);
  test_core_sendrecv(sv, rv);
}
TYPED_TEST(SendRecv, NonContig2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "v", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "v", 137, 17);
  test_core_sendrecv(sv, rv);
}

}  // namespace
