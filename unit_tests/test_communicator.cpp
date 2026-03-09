// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

using Ex = Kokkos::DefaultExecutionSpace;
using Co = KokkosComm::DefaultCommunicationSpace;

// ============================================================================
// from_raw
// ============================================================================

TEST(Communicator, from_raw_null_returns_nullopt) {
  // from_raw must return std::nullopt when passed a null communicator handle
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto result = KokkosComm::Communicator<Co, Ex>::from_raw(nullptr, Ex{});
#else
  auto result             = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_NULL, Ex{});
#endif
  ASSERT_FALSE(result.has_value());
}

TEST(Communicator, from_raw_valid_returns_communicator) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto result   = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{});
#else
  auto result             = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{});
#endif
  ASSERT_TRUE(result.has_value());
}

TEST(Communicator, from_raw_size_and_rank_are_consistent) {
  // size()/rank() must match the created-from raw communicator size/rank
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx           = test_utils::nccl::Ctx::init();
  auto comm               = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
  const int expected_size = [&nccl_ctx]() {
    int s;
    ncclCommCount(nccl_ctx.comm(), &s);
    return s;
  }();
  const int expected_rank = [&nccl_ctx]() {
    int r;
    ncclCommUserRank(nccl_ctx.comm(), &r);
    return r;
  }();
#else
  auto comm               = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
  const int expected_size = []() {
    int s;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
  }();
  const int expected_rank = []() {
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
  }();
#endif
  ASSERT_EQ(comm.size(), expected_size);
  ASSERT_EQ(comm.rank(), expected_rank);
}

// ============================================================================
// duplicate
// ============================================================================

TEST(Communicator, duplicate_from_raw_returns_communicator) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto result   = KokkosComm::Communicator<Co, Ex>::duplicate(nccl_ctx.comm(), Ex{});
#else
  auto result   = KokkosComm::Communicator<Co, Ex>::duplicate(MPI_COMM_WORLD, Ex{});
#endif
  ASSERT_TRUE(result.has_value());
}

TEST(Communicator, duplicate_preserves_size_and_rank) {
  // A duplicated communicator must have the same size and rank as the source
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
#else
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
#endif
  auto dup = original.duplicate().value();

  ASSERT_EQ(dup.size(), original.size());
  ASSERT_EQ(dup.rank(), original.rank());
}

TEST(Communicator, duplicate_produces_independent_communicator) {
  // The duplicated communicator must hold a distinct handle from the original.
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
#else
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
#endif
  auto dup = original.duplicate().value();

  ASSERT_NE(dup.comm(), original.comm());
}

// ============================================================================
// split
// ============================================================================

TEST(Communicator, split_undefined_color_returns_nullopt) {
  // A rank that passes MPI_UNDEFINED/NCCL_SPLIT_NOCOLOR as color must be excluded from the new communicator and receive
  // std::nullopt
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
  auto result   = original.split(NCCL_SPLIT_NOCOLOR, 0);
#else
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
  auto result   = original.split(MPI_UNDEFINED, 0);
#endif
  ASSERT_FALSE(result.has_value());
}

TEST(Communicator, split_same_color_groups_all_ranks) {
  // When all ranks use the same color, the resulting communicator must have the same size as the original
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto comm     = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
#else
  auto comm     = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
#endif
  auto result = comm.split(0, comm.rank());

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), comm.size());
}

TEST(Communicator, split_two_colors_produces_half_sized_communicators) {
  // With two colors assigned by parity, each sub-communicator must contain roughly half the ranks.
  // Requires an even number of ranks >= 2.
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto comm     = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
#else
  auto comm     = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
#endif
  if (comm.size() < 2 or comm.size() % 2 != 0) {
    GTEST_SKIP() << "Requires an even number of ranks >= 2 (" << comm.size() << " provided)";
  }

  const int color = comm.rank() % 2;
  auto result     = comm.split(color, comm.rank());

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), comm.size() / 2);
}

TEST(Communicator, split_key_controls_rank_ordering) {
  // When ranks split into a single group with reversed keys, the rank order inside the new communicator must be the
  // mirror of the original
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto comm     = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
#else
  auto comm     = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
#endif
  if (comm.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << comm.size() << " provided)";
  }

  // Reverse key: rank 0 gets the highest key, rank N-1 gets 0
  const int reversed_key = comm.size() - 1 - comm.rank();
  auto result            = comm.split(0, reversed_key);

  ASSERT_TRUE(result.has_value());
  // After reversal, each rank's new rank must equal its reversed_key position
  ASSERT_EQ(result->rank(), reversed_key);
}

// ============================================================================
// Move semantics
// ============================================================================

TEST(Communicator, move_constructed_communicator_is_valid) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
#else
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
#endif
  const int expected_size = original.size();
  const int expected_rank = original.rank();

  auto moved = std::move(original);

  ASSERT_EQ(moved.size(), expected_size);
  ASSERT_EQ(moved.rank(), expected_rank);
}

TEST(Communicator, move_assigned_communicator_is_valid) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(nccl_ctx.comm(), Ex{}).value();
  auto target   = KokkosComm::Communicator<Co, Ex>::duplicate(nccl_ctx.comm(), Ex{}).value();
#else
  auto original = KokkosComm::Communicator<Co, Ex>::from_raw(MPI_COMM_WORLD, Ex{}).value();
  auto target   = KokkosComm::Communicator<Co, Ex>::duplicate(MPI_COMM_WORLD, Ex{}).value();
#endif
  const int expected_size = original.size();
  const int expected_rank = original.rank();

  target = std::move(original);

  ASSERT_EQ(target.size(), expected_size);
  ASSERT_EQ(target.rank(), expected_rank);
}

}  // namespace
