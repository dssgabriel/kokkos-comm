// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <format>

#include <cuda.h>
#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/fwd.hpp>
#include "nccl_space.hpp"

namespace KokkosComm {

/*
- init_fence
- allocations
- pre_copies
- pre_comm_fence
- comm
- wait
- post-wait
*/
template <KokkosExecutionSpace ExecSpace>
class Handle<ExecSpace, Experimental::Nccl> {
 public:
  using execution_space     = ExecSpace;
  using communication_space = Experimental::Nccl;
  using communicator_type   = ncclComm_t;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
  using rank_type           = int;

  explicit Handle(const execution_space &space, communicator_type comm) : space_(space), comm_(comm) {}
  explicit Handle(communicator_type comm) : Handle(execution_space{}, comm) {}

  /// This initializes NCCL communicators for multiple GPUs within a single process.
  /// It is the simplest NCCL setup, as it does not rely on MPI, and is ideal for applications that want to use
  /// multiple GPUs without the complexity of multi-process coordination.
  ///
  /// FIXME: Do we want to allow users creating a NCCL Handle without providing the communicator?
  /// This requires us initializing it manually, which, if we do distributed initialization, is a lot more work than
  /// for initializing MPI. The implementation below does not really match what KokkosComm is trying to solve,
  /// i.e. enabling multi-node communications.
  /// Commenting it out for now.
  // Handle() {
  //   // Discover how many CUDA devices are available
  //   int n_gpus;
  //   cudaGetDeviceCount(&n_gpus);
  //   if (n_gpus == 0) {
  //     std::cerr << "KokkosComm::Handle<Cuda, Nccl>: error: no CUDA devices found\n";
  //     Kokkos::abort();
  //   }

  //   // Allocate arrays to hold our per-device resources
  //   // We need one communicator, stream, and device ID per GPU
  //   std::vector<int> dev_ids(n_gpus);
  //   std::vector<ncclComm_t> comms(n_gpus);
  //   std::vector<cudaStream_t> streams(n_gpus);

  //   // Create device list. Default: use all available devices.
  //   for (int i = 0; i < n_gpus; ++i) {
  //     dev_ids[i] = i;  // Use device i for communicator i
  //     cudaSetDevice(dev_ids[i]);
  //     cudaStreamCreate(&streams[i]);
  //   }

  //   // Initialize NCCL communicators.
  //   ncclCommInitAll(comms, n_gpus, dev_ids);
  //   // This is semantically wrong, since our Handle only remembers the communicator of the first device.
  //   return Handle(execution_space{}, comms[0]);
  // }

  auto comm() -> communicator_type & { return comm_; }
  auto comm() const -> const communicator_type & { return comm_; }
  auto space() -> execution_space & { return space_; }
  auto space() const -> const execution_space & { return space_; }

  auto size() -> rank_type {
    rank_type ret;
    ncclCommCount(comm_, &ret);
    return ret;
  }

  auto rank() -> rank_type {
    rank_type ret;
    ncclCommUserRank(comm_, &ret);
    return ret;
  }

 private:
  execution_space space_;
  communicator_type comm_;
};

}  // namespace KokkosComm
