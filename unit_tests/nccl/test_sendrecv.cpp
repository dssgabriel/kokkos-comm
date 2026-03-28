// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "../view_utils.hpp"
#include "utils.hpp"

namespace {

template <KokkosComm::KokkosView SendV, KokkosComm::KokkosView RecvV>
auto test_nccl_sendrecv(const SendV& sv, const RecvV& rv) -> void {
  auto nccl_ctx  = test_utils::nccl::Ctx::init();
  auto exec      = Kokkos::DefaultExecutionSpace{};
  auto comm      = KokkosComm::Communicator<>::from_raw(nccl_ctx.comm(), exec);
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int peer = (rank + 1) % size;

  test_utils::init_view(exec, sv);
  exec.fence();

  KokkosComm::Experimental::nccl::send(comm, sv, peer, rv, peer).wait();

  int errs = test_utils::count_errors(rv);
  ASSERT_EQ(errs, 0);
}

template <typename T>
class NcclSendrecv : public ::testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;
TYPED_TEST_SUITE(NcclSendrecv, ScalarTypes);

TYPED_TEST(NcclSendrecv, Contig1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "sv", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "rv", 1013);
  test_nccl_sendrecv(sv, rv);
}
TYPED_TEST(NcclSendrecv, NonContig1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "sv", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "rv", 1013);
  test_nccl_sendrecv(sv, rv);
}
TYPED_TEST(NcclSendrecv, Contig2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "sv", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "rv", 137, 17);
  test_nccl_sendrecv(sv, rv);
}
TYPED_TEST(NcclSendrecv, NonContig2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "sv", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "rv", 137, 17);
  test_nccl_sendrecv(sv, rv);
}

}  // namespace
