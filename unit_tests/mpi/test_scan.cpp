//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

namespace {

template <typename T>
class Scan : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(Scan, ScalarTypes);

template <typename Scalar>
void test_inclusive_scan_0d() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar> rv("rv");

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int) { sv() = rank; });

  KokkosComm::mpi::inclusive_scan(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_SUM, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(int, int &lsum) { lsum += (rv() != rank * (rank + 1) / 2); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Scan, 0D) { test_inclusive_scan_0d<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_inclusive_scan_1d_contig() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int nContrib = 10;

  Kokkos::View<Scalar *> sv("sv", nContrib);
  Kokkos::View<Scalar *> rv("rv", nContrib);

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int const i) { sv(i) = rank + i; });

  KokkosComm::mpi::inclusive_scan(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_SUM, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(int i, int &lsum) { lsum += (rv(i) != (rank + i) * (rank + i + 1) / 2 - i * (i - 1) / 2); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Scan, 1D_contig) { test_inclusive_scan_1d_contig<typename TestFixture::Scalar>(); }

}  // namespace
