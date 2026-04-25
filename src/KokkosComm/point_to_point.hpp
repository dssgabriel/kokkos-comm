// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core_fwd.hpp>

#include "fwd.hpp"
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "mpi/mpi_space.hpp"
#include "mpi/communicator.hpp"
#include "mpi/request.hpp"
#include "mpi/send.hpp"
#include "mpi/recv.hpp"
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/nccl_space.hpp"
#include "nccl/communicator.hpp"
#include "nccl/request.hpp"
#include "nccl/send.hpp"
#include "nccl/recv.hpp"
#endif

namespace KokkosComm {
namespace Impl {

#if defined(KOKKOSCOMM_ENABLE_MPI)
template <KokkosExecutionSpace Exec, KokkosView SendV>
struct Send<MpiSpace, Exec, SendV> {
  static auto execute(Communicator<MpiSpace, Exec>& comm, const SendV& sv, int dst) -> Request<MpiSpace> {
    return mpi::isend(comm, sv, dst, mpi::Impl::P2P_TAG);
  }
};
template <KokkosExecutionSpace Exec, KokkosView RecvV>
struct Recv<MpiSpace, Exec, RecvV> {
  static auto execute(Communicator<MpiSpace, Exec>& comm, const RecvV& rv, int src) -> Request<MpiSpace> {
    return mpi::irecv(comm, rv, src, mpi::Impl::P2P_TAG);
  }
};
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <KokkosView SendV>
struct Send<Experimental::NcclSpace, Kokkos::Cuda, SendV> {
  static auto execute(Communicator<Experimental::NcclSpace, Kokkos::Cuda>& comm, const SendV& sv, int dst)
      -> Request<Experimental::NcclSpace> {
    return Experimental::nccl::send(comm, sv, dst);
  }
};
template <KokkosView RecvV>
struct Recv<Experimental::NcclSpace, Kokkos::Cuda, RecvV> {
  static auto execute(Communicator<Experimental::NcclSpace, Kokkos::Cuda>& comm, const RecvV& rv, int src)
      -> Request<Experimental::NcclSpace> {
    return Experimental::nccl::recv(comm, rv, src);
  }
};
#endif

}  // namespace Impl

/// @brief Initiates a non-blocking send operation.
/// @tparam Comm A Communication space type.
/// @tparam Exec A Kokkos execution space type.
/// @tparam SendV A Kokkos View type.
/// @param comm The communicator handle associated with the operation.
/// @param sv The View to send.
/// @param dst The destination rank.
/// @returns A Request object representing the non-blocking send operation.
template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView SendV>
auto send(Communicator<Comm, Exec>& comm, const SendV& sv, int dst) -> Request<Comm> {
  return Impl::Send<Comm, Exec, SendV>::execute(comm, sv, dst);
}

/// @brief Initiates a non-blocking receive operation.
/// @tparam Comm A Communication space type.
/// @tparam Exec A Kokkos execution space type.
/// @tparam RecvV A Kokkos View type.
/// @param comm The communicator handle associated with the operation.
/// @param rv The View to receive into.
/// @param src The source rank.
/// @returns A Request object representing the non-blocking receive operation.
template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView RecvV>
auto recv(Communicator<Comm, Exec>& comm, const RecvV& rv, int src) -> Request<Comm> {
  return Impl::Recv<Comm, Exec, RecvV>::execute(comm, rv, src);
}

}  // namespace KokkosComm
