// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

template <typename T>
class AllReduce : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(AllReduce, ScalarTypes);

template <typename Scalar>
auto allreduce_0d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, CommSpace> h(nccl_ctx.comm());

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar> rv("rv");

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(ExecSpace(), sv.extent(0)), KOKKOS_LAMBDA(int) { sv() = h.rank(); });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::allreduce(h, sv, rv, KokkosComm::Sum);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(int, int &lsum) { lsum += (rv() != h.size() * (h.size() - 1) / 2); }, errs);
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto allreduce_contig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, CommSpace> h(nccl_ctx.comm());

  int n_contrib = 10;
  Kokkos::View<Scalar *> sv("sv", n_contrib);
  Kokkos::View<Scalar *> rv("rv", n_contrib);

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(ExecSpace(), sv.extent(0)), KOKKOS_LAMBDA(int i) { sv(i) = h.rank() + i; });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::allreduce(h, sv, rv, KokkosComm::Sum);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(int i, int &lsum) { lsum += (rv(i) != h.size() * (h.size() - 1) / 2 + h.size() * i); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(AllReduce, 0D) { allreduce_0d<typename TestFixture::Scalar>(); }

TYPED_TEST(AllReduce, Contiguous1D) { allreduce_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
