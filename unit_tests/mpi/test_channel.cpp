//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
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
#include <type_traits>

#include "KokkosComm/KokkosComm.hpp"

namespace {

using namespace KokkosComm::mpi;

template <typename T>
class ChannelSendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(ChannelSendRecv, ScalarTypes);

template <typename Scalar>
void test_channel_send_recv() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int dest_rank = (rank + 1) % size;         // send to next rank
  const int src_rank  = (rank - 1 + size) % size;  // recv from prev rank
  const int tag       = 42;

  KokkosComm::Channel<> channel(dest_rank, src_rank, tag, MPI_COMM_WORLD);

  const int N = 10;

  // Create host views
  Kokkos::View<Scalar*, Kokkos::HostSpace> send_host("send_host", N);
  Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("recv_host", N);

  for (int i = 0; i < N; i++) send_host(i) = static_cast<Scalar>(rank * N + i);

  channel.sendinit(send_host);
  channel.recvinit(recv_host);

  channel.start();
  channel.wait();

  int errs = 0;
  for (int i = 0; i < N; i++) {
    const Scalar expected = static_cast<Scalar>(src_rank * N + i);
    if (recv_host(i) != expected) {
      errs++;
    }
  }
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(ChannelSendRecv, 1D_contig_standard) { test_channel_send_recv<typename TestFixture::Scalar>(); }

}  // namespace
