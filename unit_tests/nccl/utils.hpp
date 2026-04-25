// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <memory>
#include <mutex>

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

namespace test_utils::nccl {

class Ctx {
 public:
  Ctx(const Ctx&) = delete;
  auto operator=(const Ctx&) -> Ctx& = delete;
  Ctx(Ctx&&) = delete;
  auto operator=(Ctx&&) -> Ctx& = delete;

  ~Ctx();

  static auto init(bool verbose = true) -> void;
  static auto fini() -> void;
  static auto get() -> Ctx&;

  auto comm() const -> ncclComm_t;
  auto stream() const -> cudaStream_t;
  auto size() const -> int;
  auto rank() const -> int;
  auto device() const -> int;

 private:
  Ctx(ncclComm_t comm,
      cudaStream_t stream,
      int dev,
      int n_ranks,
      int my_rank);

  ncclComm_t comm_{nullptr};
  cudaStream_t stream_{nullptr};
  int dev_{-1};
  int n_ranks_{0};
  int my_rank_{0};

  static std::unique_ptr<Ctx> instance_;
  static std::once_flag init_flag_;
};

}  // namespace test_utils::nccl
