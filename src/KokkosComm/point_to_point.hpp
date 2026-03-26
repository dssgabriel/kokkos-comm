// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core_fwd.hpp>

#include "concepts.hpp"
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

}  // namespace Impl

template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView SendV>
auto send(Communicator<Comm, Exec>& comm, const SendV& sv, int dst) -> Request<Comm> {
  return Impl::Send<Comm, Exec, SendV>::execute(comm, sv, dst);
}

template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView RecvV>
auto recv(Communicator<Comm, Exec>& comm, const RecvV& rv, int src) -> Request<Comm> {
  return Impl::Recv<Comm, Exec, RecvV>::execute(comm, rv, src);
}

}  // namespace KokkosComm
