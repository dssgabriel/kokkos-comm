// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Experimental::Nccl;

template <typename T>
class Broadcast : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(Broadcast, ScalarTypes);

template <typename Scalar>
auto broadcast_0d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  KokkosComm::Handle h(ExecSpace(), nccl_ctx.comm());
  int root = 0;

  Kokkos::View<Scalar> v("v");
  if (h.rank() == root) {
    // Prepare broadcast view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(ExecSpace(), 0, v.extent(0)), KOKKOS_LAMBDA(int) { v() = size; });
  }
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::broadcast(h, v, root);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(int, int &lsum) { lsum += v() != size; }, errs);
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto broadcast_inplace_contig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, CommSpace> h(nccl_ctx.comm());
  int root = 0;

  Kokkos::View<Scalar *> v("v", 100);
  if (rank == root) {
    // Prepare broadcast view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(ExecSpace(), 0, v.extent(0)), KOKKOS_LAMBDA(int i) { v(i) = size + i; });
  }
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::broadcast(h, v, root);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(int i, int &lsum) { lsum += (v(i) != size + i); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Broadcast, InPlace0D) { broadcast_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(Broadcast, InPlaceContiguous1D) { broadcast_inplace_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
