// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdint>

#include <gtest/gtest.h>
#include <nccl.h>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Experimental::Nccl;

template <typename T>
class P2P : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<float, double, int, unsigned, int64_t, uint64_t>;

TYPED_TEST_SUITE(P2P, ScalarTypes);

template <typename Scalar>
auto p2p_1d_contig() -> void {
  Kokkos::View<Scalar *> a("a", 1000);

  auto nccl_ctx = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, CommSpace> h(nccl_ctx.comm());
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

  auto nccl_ctx = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, CommSpace> h(nccl_ctx.comm());
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
