// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

namespace {

enum struct LogLevel {
  FATAL,
  ERROR,
  WARN,
  INFO,
  TRACE,
};

using namespace std::string_view_literals;
constexpr std::array level_txt{"FATAL"sv, "ERROR"sv, "WARNING"sv, "INFO"sv, "TRACE"sv};

#define KC_LOG(lvl, ...) \
  fmt::println("[{}] {}:{}: {}", level_txt[static_cast<int>(lvl)], __FILE__, __LINE__, fmt::format(__VA_ARGS__))

#define KC_FATAL(...) (KC_LOG(LogLevel::FATAL, __VA_ARGS__), std::exit(EXIT_FAILURE))

#define KC_ERROR(...) KC_LOG(LogLevel::ERROR, __VA_ARGS__)

#define KC_WARN(...) KC_LOG(LogLevel::WARN, __VA_ARGS__)

#define KC_INFO(...) KC_LOG(LogLevel::INFO, __VA_ARGS__)

#define KC_TRACE(...) KC_LOG(LogLevel::TRACE, __VA_ARGS__)

#define KC_CHECK(expr, ...) ((expr) ? void(0) : KC_FATAL(__VA_ARGS__))

#define KC_MPI_CHECK(expr)                                                                            \
  ([&]() {                                                                                            \
    int kc_res_ = (expr);                                                                             \
    return kc_res_ == MPI_SUCCESS ? void(0) : KC_FATAL("MPI check failed: `" #expr "`: {}", kc_res_); \
  }())

#define KC_NCCL_CHECK(expr)                                                                                      \
  ([&]() {                                                                                                       \
    ncclResult_t kc_res_ = (expr);                                                                               \
    return kc_res_ == ncclSuccess ? void(0)                                                                      \
                                  : KC_FATAL("NCCL check failed: `" #expr "`: {}", ncclGetErrorString(kc_res_)); \
  }())

#define KC_CUDA_CHECK(expr)                                                                                      \
  ([&]() {                                                                                                       \
    cudaError_t kc_res_ = (expr);                                                                                \
    return kc_res_ == cudaSuccess ? void(0)                                                                      \
                                  : KC_FATAL("CUDA check failed: `" #expr "`: {}", cudaGetErrorString(kc_res_)); \
  }())

[[nodiscard]] auto get_local_rank(MPI_Comm comm, int my_rank) -> int {
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &node_comm);

  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);

  MPI_Comm_free(&node_comm);
  return node_rank;
}

}  // namespace

namespace test_utils::nccl {

class Ctx {
 public:
  static auto init() -> Ctx {
    int flag;
    MPI_Initialized(&flag);
    KC_CHECK(flag == true, "MPI is not initialized");

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Retrieve rank info
    int n_ranks, my_rank;
    MPI_Comm_size(mpi_comm, &n_ranks);
    MPI_Comm_rank(mpi_comm, &my_rank);
    KC_INFO("P%d/%d - MPI initialized", n_ranks, my_rank);
    int local_rank = get_local_rank(mpi_comm, my_rank);

    int n_gpus;
    KC_CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    KC_INFO("P%d found %d CUDA devices", my_rank, n_gpus);

    KC_CHECK(local_rank <= n_gpus, "P%d needs GPU %d but only %d devices available", my_rank, local_rank, n_gpus);
    KC_CUDA_CHECK(cudaSetDevice(local_rank));
    KC_INFO("P%d assigned to CUDA device %d", my_rank, local_rank);

    // Get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId nccl_id;
    if (my_rank == 0) {
      ncclGetUniqueId(&nccl_id);
    }
    KC_MPI_CHECK(MPI_Bcast(&nccl_id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, mpi_comm));

    // NCCL comm configuration
    ncclConfig_t nccl_cfg = NCCL_CONFIG_INITIALIZER;
    // Always non-blocking communicator
    nccl_cfg.blocking = 0;

    // Initialize NCCL communicator
    ncclComm_t nccl_comm;
    KC_NCCL_CHECK(ncclCommInitRankConfig(&nccl_comm, n_ranks, nccl_id, my_rank, &nccl_cfg));

    return Ctx(nccl_comm, n_ranks, my_rank);
  }

  ~Ctx() { KC_NCCL_CHECK(ncclCommDestroy(comm_)); }
  Ctx(const Ctx &)                     = delete;
  auto operator=(const Ctx &) -> Ctx & = delete;
  Ctx(Ctx &&)                          = delete;
  auto operator=(Ctx &&) -> Ctx      & = delete;

  auto comm() -> ncclComm_t & { return comm_; }
  auto n_ranks() -> int { return n_ranks_; }
  auto my_rank() -> int { return my_rank_; }

 private:
  explicit Ctx(ncclComm_t comm, int n_ranks, int my_rank) : comm_(comm), n_ranks_(n_ranks), my_rank_(my_rank) {}

  ncclComm_t comm_;
  int n_ranks_;
  int my_rank_;
};

}  // namespace test_utils::nccl
