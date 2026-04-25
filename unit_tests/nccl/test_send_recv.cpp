// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "../view_utils.hpp"
#include "utils.hpp"

namespace {

template <KokkosComm::KokkosView View>
auto test_nccl_send_recv(const View& v) -> void {
  auto nccl_ctx  = test_utils::nccl::Ctx::init();
  auto exec      = Kokkos::DefaultExecutionSpace{};
  auto comm      = KokkosComm::Communicator<>::from_raw(nccl_ctx.comm(), exec);
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int src = 0;
  const int dst = 1;

  if (rank == src) {
    test_utils::init_view(exec, v);
    exec.fence();

    KokkosComm::Experimental::nccl::send(comm, v, dst).wait();
  } else if (rank == dst) {
    KokkosComm::Experimental::nccl::recv(comm, v, src).wait();

    int errs = test_utils::count_errors(v);
    ASSERT_EQ(errs, 0);
  }
}

template <typename T>
class NcclSendRecv : public ::testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;
TYPED_TEST_SUITE(NcclSendRecv, ScalarTypes);

TYPED_TEST(NcclSendRecv, Contig1D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "v", 1013);
  test_nccl_send_recv(v);
}
TYPED_TEST(NcclSendRecv, NonContig1D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "v", 1013);
  test_nccl_send_recv(v);
}
TYPED_TEST(NcclSendRecv, Contig2D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "v", 137, 17);
  test_nccl_send_recv(v);
}
TYPED_TEST(NcclSendRecv, NonContig2D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "v", 137, 17);
  test_nccl_send_recv(v);
}
TYPED_TEST(NcclSendRecv, Contig3D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::Contig{}, "v", 13, 10, 7);
  test_nccl_send_recv(v);
}
TYPED_TEST(NcclSendRecv, NonContig3D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::NonContig{}, "v", 13, 10, 7);
  test_nccl_send_recv(v);
}

}  // namespace
