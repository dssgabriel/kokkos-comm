// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

using CommSpace   = KokkosComm::DefaultCommunicationSpace;
using RequestType = KokkosComm::Request<CommSpace>;

static_assert(!std::is_copy_constructible_v<RequestType>);
static_assert(!std::is_copy_assignable_v<RequestType>);
static_assert(std::is_nothrow_move_constructible_v<RequestType>);
static_assert(std::is_nothrow_move_assignable_v<RequestType>);

[[nodiscard]] auto null_request() -> RequestType::request_type {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  return nullptr;
#else
  return MPI_REQUEST_NULL;
#endif
}

[[nodiscard]] auto make_request() -> RequestType {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto raw_comm = test_utils::NcclCtx::get().comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto comm = KokkosComm::Communicator<>::from_raw(raw_comm);
  Kokkos::View<int*> buffer("request_test_buffer", 1);
  return KokkosComm::Experimental::broadcast(comm, buffer, 0);
}

TEST(Request, default_constructed_request_is_inactive) {
  RequestType request;

  ASSERT_EQ(request.request(), null_request());
}

TEST(Request, move_construction_transfers_ownership) {
  auto source                 = make_request();
  const auto expected_request = source.request();
  bool callback_executed      = false;
  source.add_callback([&callback_executed]() { callback_executed = true; });

  RequestType target(std::move(source));

  ASSERT_EQ(source.request(), null_request());
  ASSERT_EQ(target.request(), expected_request);

  target.wait();
  ASSERT_TRUE(callback_executed);
}

TEST(Request, move_assignment_transfers_ownership) {
  auto source                 = make_request();
  const auto expected_request = source.request();
  bool callback_executed      = false;
  source.add_callback([&callback_executed]() { callback_executed = true; });

  RequestType target;
  target = std::move(source);

  ASSERT_EQ(source.request(), null_request());
  ASSERT_EQ(target.request(), expected_request);

  target.wait();
  ASSERT_TRUE(callback_executed);
}

TEST(Request, self_move_assignment_is_safe) {
  auto request                = make_request();
  const auto expected_request = request.request();

  request = std::move(request);

  ASSERT_EQ(request.request(), expected_request);
  request.wait();
}

}  // namespace
