// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <atomic>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

using RequestType = KokkosComm::Request<KokkosComm::DefaultCommunicationSpace>;

class RequestDeathTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

static_assert(!std::is_copy_constructible_v<RequestType>);
static_assert(!std::is_copy_assignable_v<RequestType>);
static_assert(std::is_nothrow_move_constructible_v<RequestType>);
static_assert(std::is_nothrow_move_assignable_v<RequestType>);

template <class Request, class View, class MakeProgress>
auto expect_completion_through_test(Request& request, const View& view, MakeProgress&& make_progress) -> void {
  const auto initial_use_count = view.use_count();
  std::vector<int> callback_order;
  request.extend_view_lifetime(view);
  EXPECT_GT(view.use_count(), initial_use_count);
  request.add_callback([&] {
    EXPECT_GT(view.use_count(), initial_use_count);
    callback_order.push_back(1);
  });
  request.add_callback([&] {
    EXPECT_GT(view.use_count(), initial_use_count);
    callback_order.push_back(2);
  });

  EXPECT_FALSE(request.test());
  EXPECT_TRUE(callback_order.empty());
  std::forward<MakeProgress>(make_progress)();
  while (!request.test()) {
  }

  EXPECT_EQ(callback_order, (std::vector<int>{1, 2}));
  EXPECT_EQ(view.use_count(), initial_use_count);
  EXPECT_TRUE(request.test());
  request.wait();
  EXPECT_EQ(callback_order, (std::vector<int>{1, 2}));
  EXPECT_EQ(view.use_count(), initial_use_count);
}

TEST(Request, EmptyCompletionIsIdempotent) {
  RequestType request;
  EXPECT_TRUE(request.test());
  request.wait();
  KokkosComm::wait(request);
  EXPECT_TRUE(KokkosComm::test(request));
  KokkosComm::wait(std::move(request));
}

TEST(Request, MovesTransferCallbacksAndManagedAllocations) {
  Kokkos::View<int*, Kokkos::HostSpace> view("retained", 1);
  const auto initial_count = view.use_count();
  std::vector<int> order;
  RequestType destination;
  {
    RequestType source;
    source.extend_view_lifetime(view);
    source.add_callback([&] { order.push_back(1); });
    source.add_callback([&] {
      EXPECT_GT(view.use_count(), initial_count);
      order.push_back(2);
    });
    EXPECT_GT(view.use_count(), initial_count);
    RequestType moved(std::move(source));
    EXPECT_TRUE(source.test());
    source.wait();
    destination = std::move(moved);
    EXPECT_TRUE(moved.test());
  }
  EXPECT_TRUE(order.empty());
  EXPECT_GT(view.use_count(), initial_count);
  auto& alias = destination;
  destination = std::move(alias);
  EXPECT_TRUE(destination.test());
  EXPECT_EQ(order, (std::vector<int>{1, 2}));
  EXPECT_EQ(view.use_count(), initial_count);
  destination.wait();
  EXPECT_TRUE(destination.test());
  EXPECT_EQ(order.size(), 2u);
  RequestType next;
  next.add_callback([&] { order.push_back(3); });
  destination = std::move(next);
  destination.wait();
  EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
}

TEST(Request, UnmanagedAllocationsRemainCallerOwned) {
  int storage = 42;
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> view(&storage, 1);
  RequestType request;
  request.extend_view_lifetime(view);
  EXPECT_TRUE(request.test());
  EXPECT_EQ(view.use_count(), 0);
  EXPECT_EQ(storage, 42);
}

#if !defined(KOKKOSCOMM_ABORT_ON_ERROR) || defined(KOKKOSCOMM_ENABLE_NCCL)
TEST_F(RequestDeathTest, DestructionWithPendingCallbacksDies) {
  EXPECT_DEATH(
      {
        RequestType request;
        request.add_callback([] {});
      },
      "destruction requires completion"
  );
}

TEST_F(RequestDeathTest, MoveAssignmentOverPendingCallbacksDies) {
  EXPECT_DEATH(
      {
        RequestType destination;
        destination.add_callback([] {});
        RequestType source;
        destination = std::move(source);
      },
      "move assignment requires an inactive destination"
  );
}

TEST_F(RequestDeathTest, EmptyCallbackIsRejected) {
  EXPECT_DEATH(
      {
        RequestType request;
        request.add_callback({});
      },
      "empty callback"
  );
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
TEST_F(RequestDeathTest, ThrowingCallbackDies) {
  EXPECT_DEATH(
      {
        RequestType request;
        request.add_callback([] { throw 42; });
        request.wait();
      },
      "completion callback threw an exception"
  );
}
#endif
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)

TEST(Request, ActiveMovesAndVectorRelocationPreserveCompletion) {
  Kokkos::Cuda space;
  Kokkos::View<int*, Kokkos::CudaSpace> view("request_data", 1);
  Kokkos::deep_copy(space, view, 42);
  RequestType destination;
  RequestType::request_type native;
  int callbacks = 0;
  // Registration before event capture is used by the receive/reduce implementations.
  {
    RequestType source;
    source.extend_view_lifetime(view);
    source.add_callback([&] { ++callbacks; });
    source.capture_stream_state(space.cuda_stream());
    native = source.request();
    RequestType moved(std::move(source));
    EXPECT_EQ(source.request(), nullptr);
    EXPECT_TRUE(source.test());
    EXPECT_EQ(moved.request(), native);
    std::vector<RequestType> requests;
    requests.reserve(1);
    requests.push_back(std::move(moved));
    requests.emplace_back();
    destination = std::move(requests.front());
    EXPECT_EQ(requests.front().request(), nullptr);
  }
  auto& alias = destination;
  destination = std::move(alias);
  EXPECT_EQ(destination.request(), native);
  destination.wait();
  EXPECT_EQ(destination.request(), nullptr);
  EXPECT_EQ(callbacks, 1);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  EXPECT_EQ(host(0), 42);
  EXPECT_TRUE(destination.test());
  EXPECT_EQ(callbacks, 1);
  destination.capture_stream_state(space.cuda_stream());
  while (!destination.test()) {
  }
  EXPECT_EQ(destination.request(), nullptr);
}

TEST(Request, IncompleteTestPreservesEventAndCallbacks) {
  Kokkos::Cuda space;
  std::atomic<bool> release{false};
  ASSERT_EQ(
      cudaLaunchHostFunc(
          space.cuda_stream(),
          [](void* flag) {
            auto& ready = *static_cast<std::atomic<bool>*>(flag);
            while (!ready.load(std::memory_order_acquire)) {
            }
          },
          &release
      ),
      cudaSuccess
  );
  RequestType request;
  int callbacks = 0;
  request.add_callback([&] { ++callbacks; });
  request.capture_stream_state(space.cuda_stream());
  auto event = request.request();
  EXPECT_FALSE(request.test());
  EXPECT_EQ(request.request(), event);
  EXPECT_EQ(callbacks, 0);
  release.store(true, std::memory_order_release);
  request.wait();
  EXPECT_EQ(request.request(), nullptr);
  EXPECT_EQ(callbacks, 1);
}

TEST(Request, SuccessfulTestRetiresCallbacksAndManagedView) {
  Kokkos::Cuda space;
  Kokkos::View<int*, Kokkos::CudaSpace> view("poll_data", 1);
  std::atomic<bool> release{false};
  ASSERT_EQ(
      cudaLaunchHostFunc(
          space.cuda_stream(),
          [](void* flag) {
            auto& ready = *static_cast<std::atomic<bool>*>(flag);
            while (!ready.load(std::memory_order_acquire)) {
            }
          },
          &release
      ),
      cudaSuccess
  );
  Kokkos::deep_copy(space, view, 42);
  RequestType request;
  request.capture_stream_state(space.cuda_stream());
  expect_completion_through_test(request, view, [&] { release.store(true, std::memory_order_release); });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  EXPECT_EQ(host(0), 42);
}

TEST_F(RequestDeathTest, DestroyingActiveRequestDies) {
  Kokkos::Cuda space;
  auto request = std::make_unique<RequestType>();
  request->capture_stream_state(space.cuda_stream());
  EXPECT_DEATH(request.reset(), "destruction requires completion");
  request->wait();
}

TEST_F(RequestDeathTest, AssigningOverActiveRequestDies) {
  Kokkos::Cuda space;
  RequestType request;
  request.capture_stream_state(space.cuda_stream());
  EXPECT_DEATH(
      {
        RequestType empty;
        request = std::move(empty);
      },
      "move assignment requires an inactive destination"
  );
  request.wait();
}

TEST_F(RequestDeathTest, CapturingAnActiveRequestDies) {
  Kokkos::Cuda space;
  RequestType request;
  request.capture_stream_state(space.cuda_stream());
  EXPECT_DEATH(request.capture_stream_state(space.cuda_stream()), "capture_stream_state: request is active");
  request.wait();
}

#else

static_assert(!std::is_constructible_v<RequestType, MPI_Request&>);
static_assert(std::is_nothrow_constructible_v<RequestType, MPI_Request&&>);

TEST(Request, ActiveMovesAndVectorRelocationPreserveCompletion) {
  int received    = -1;
  int sent        = 42;
  MPI_Request raw = MPI_REQUEST_NULL;
  ASSERT_EQ(MPI_Irecv(&received, 1, MPI_INT, 0, 123, MPI_COMM_SELF, &raw), MPI_SUCCESS);
  RequestType destination;
  RequestType::request_type native;
  int callbacks = 0;
  // Regression scenario from PR #282: destroy the source before completing the destination.
  // https://github.com/kokkos/kokkos-comm/pull/282
  {
    RequestType source(std::move(raw));
    EXPECT_EQ(raw, MPI_REQUEST_NULL);
    source.add_callback([&] { ++callbacks; });
    EXPECT_FALSE(source.test());  // No matching send has been posted.
    EXPECT_EQ(callbacks, 0);
    native = source.request();  // Borrowed identity comparison only.
    RequestType moved(std::move(source));
    EXPECT_EQ(source.request(), MPI_REQUEST_NULL);
    EXPECT_EQ(moved.request(), native);
    EXPECT_TRUE(source.test());
    std::vector<RequestType> requests;
    requests.reserve(1);
    requests.push_back(std::move(moved));
    requests.emplace_back();  // Forces relocation of the active receive.
    EXPECT_EQ(moved.request(), MPI_REQUEST_NULL);
    destination = std::move(requests.front());
    EXPECT_EQ(requests.front().request(), MPI_REQUEST_NULL);
  }
  auto& alias = destination;
  destination = std::move(alias);
  EXPECT_EQ(destination.request(), native);
  RequestType send;
  ASSERT_EQ(MPI_Isend(&sent, 1, MPI_INT, 0, 123, MPI_COMM_SELF, send.request_ptr()), MPI_SUCCESS);
  send.wait();
  destination.wait();
  EXPECT_EQ(destination.request(), MPI_REQUEST_NULL);
  EXPECT_EQ(received, sent);
  EXPECT_EQ(callbacks, 1);
  EXPECT_TRUE(destination.test());
  destination.wait();
  EXPECT_EQ(callbacks, 1);
}

TEST(Request, SuccessfulTestRetiresCallbacksAndManagedView) {
  Kokkos::View<int*, Kokkos::HostSpace> received("poll_recv", 1);
  RequestType request;
  ASSERT_EQ(MPI_Irecv(received.data(), 1, MPI_INT, 0, 124, MPI_COMM_SELF, request.request_ptr()), MPI_SUCCESS);
  int sent = 17;
  expect_completion_through_test(request, received, [&] {
    RequestType send;
    EXPECT_EQ(MPI_Isend(&sent, 1, MPI_INT, 0, 124, MPI_COMM_SELF, send.request_ptr()), MPI_SUCCESS);
    send.wait();
  });
  EXPECT_EQ(received(0), sent);
}

#if !defined(KOKKOSCOMM_ABORT_ON_ERROR)
class RequestMpiDeathTest : public RequestDeathTest {
 protected:
  void SetUp() override {
    RequestDeathTest::SetUp();
    ASSERT_EQ(MPI_Irecv(&received_, 1, MPI_INT, 0, 125, MPI_COMM_SELF, request_.request_ptr()), MPI_SUCCESS);
  }

  auto complete_request() -> void {
    int sent         = 73;
    MPI_Request send = MPI_REQUEST_NULL;
    ASSERT_EQ(MPI_Isend(&sent, 1, MPI_INT, 0, 125, MPI_COMM_SELF, &send), MPI_SUCCESS);
    ASSERT_EQ(MPI_Wait(&send, MPI_STATUS_IGNORE), MPI_SUCCESS);
    request_.wait();
    EXPECT_EQ(received_, sent);
  }

  int received_ = -1;
  RequestType request_;
};

TEST_F(RequestMpiDeathTest, DestroyingActiveRequestDies) {
  auto request = std::make_unique<RequestType>(std::move(request_));
  EXPECT_DEATH(request.reset(), "destruction requires completion");
  complete_request();
  request->wait();
}

TEST_F(RequestMpiDeathTest, AssigningOverActiveRequestDies) {
  auto request = std::make_unique<RequestType>(std::move(request_));
  EXPECT_DEATH(
      {
        RequestType empty;
        *request = std::move(empty);
      },
      "move assignment requires an inactive destination"
  );
  complete_request();
  request->wait();
}
#endif

#endif
