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
template <KokkosExecutionSpace Ex, KokkosView SendV>
struct Send<MpiSpace, Ex, SendV> {
  static auto execute(Communicator<MpiSpace, Ex>& comm, const SendV& sv, int dst) -> Request<MpiSpace> {
    return mpi::isend(comm.exec(), sv, dst, mpi::Impl::P2P_TAG, comm.comm());
  }
};
template <KokkosExecutionSpace Ex, KokkosView RecvV>
struct Recv<MpiSpace, Ex, RecvV> {
  static auto execute(Communicator<MpiSpace, Ex>& comm, const RecvV& rv, int src) -> Request<MpiSpace> {
    return mpi::irecv(comm.exec(), rv, src, mpi::Impl::P2P_TAG, comm.comm());
  }
};
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <KokkosView SendV>
struct Send<Experimental::NcclSpace, Kokkos::Cuda, SendV> {
  static auto execute(Communicator<Experimental::NcclSpace, Kokkos::Cuda>& comm, const SendV& sv, int dst)
      -> Request<Experimental::NcclSpace> {
    return nccl::send(comm.exec(), sv, dst, comm.comm());
  }
};
template <KokkosView RecvV>
struct Recv<Experimental::NcclSpace, Kokkos::Cuda, RecvV> {
  static auto execute(Communicator<Experimental::NcclSpace, Kokkos::Cuda>& comm, const RecvV& rv, int src)
      -> Request<Experimental::NcclSpace> {
    return nccl::recv(comm.exec(), rv, src, comm.comm());
  }
};
#endif

}  // namespace Impl

template <CommunicationSpace Co, KokkosExecutionSpace Ex, KokkosView SendV>
auto send(Communicator<Co, Ex>& comm, const SendV& sv, int dst) -> Request<Co> {
  return Impl::Send<Co, Ex, SendV>::execute(comm, sv, dst);
}

template <CommunicationSpace Co, KokkosExecutionSpace Ex, KokkosView RecvV>
auto recv(Communicator<Co, Ex>& comm, const RecvV& rv, int src) -> Request<Co> {
  return Impl::Recv<Co, Ex, RecvV>::execute(comm, rv, src);
}

}  // namespace KokkosComm
