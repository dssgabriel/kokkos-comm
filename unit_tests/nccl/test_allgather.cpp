// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Experimental::Nccl;

template <typename T>
class AllGather : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(AllGather, ScalarTypes);

template <typename Scalar>
auto allgather_0d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  KokkosComm::Handle h(ExecSpace(), nccl_ctx.comm());

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar *> rv("rv", h.size());

  // Fill send view, 1 element per sender: their rank
  Kokkos::parallel_for(
      Kokkos::RangePolicy(ExecSpace(), 0, sv.extent(0)), KOKKOS_LAMBDA(int) { sv() = h.rank(); });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::allgather(h, sv, rv);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(int &src, int &lsum) { lsum += rv(src) != src; }, errs);
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto allgather_contig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  KokkosComm::Handle h(ExecSpace(), nccl_ctx.comm());

  const int n_contrib = 100;
  Kokkos::View<Scalar *> sv("sv", n_contrib);
  Kokkos::View<Scalar *> rv("rv", h.size() * n_contrib);

  // Fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int i) { sv(i) = h.rank() + i; });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::allgather(h, sv, rv);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(int &i, int &lsum) {
        const int src = i / n_contrib;
        const int j   = i % n_contrib;
        lsum += rv(i) != src + j;
      },
      errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(AllGather, 0D) { allgather_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(AllGather, Contiguous1D) { allgather_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
