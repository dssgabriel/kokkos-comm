// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "mpi_space.hpp"
#include "communicator.hpp"
#include "request.hpp"
#include "comm_mode.hpp"

#include "impl/pack_traits.hpp"
#include "impl/tags.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace Exec, KokkosView SendV, CommunicationMode SendMode>
auto isend(Communicator<MpiSpace, Exec>& comm, const SendV& sv, int dst, int tag, SendMode) -> Request<MpiSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::isend");

  auto mpi_isend_fn = [dst, tag](void* buf, int count, MPI_Datatype dtype, MPI_Request* req_ptr, MPI_Comm mpi_comm) {
    if constexpr (std::is_same_v<SendMode, CommModeStandard>) {
      MPI_Isend(buf, count, dtype, dst, tag, mpi_comm, req_ptr);
    } else if constexpr (std::is_same_v<SendMode, CommModeReady>) {
      MPI_Irsend(buf, count, dtype, dst, tag, mpi_comm, req_ptr);
    } else if constexpr (std::is_same_v<SendMode, CommModeSynchronous>) {
      MPI_Issend(buf, count, dtype, dst, tag, mpi_comm, req_ptr);
    } else {
      static_assert(std::is_void_v<SendMode>, "unexpected communication mode");
    }
  };

  const auto exec = comm.exec();
  Request<MpiSpace> req;
  if (is_contiguous(sv)) {
    exec.fence("fence before MPI_Isend");
    mpi_isend_fn(data_handle(sv), span(sv), datatype_for<MpiSpace>(sv), req.request_ptr(), comm.comm());
  } else {
    using Packer = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv = Packer::pack(exec, "pckd_sv", sv);
    exec.fence("fence packed allocation before MPI_Isend");
    mpi_isend_fn(data_handle(pckd_sv.view), pckd_sv.count, pckd_sv.datatype, req.request_ptr(), comm.comm());
    req.extend_view_lifetime(pckd_sv.view);
  }
  req.extend_view_lifetime(sv);

  Kokkos::Tools::popRegion();
  return req;
}
template <KokkosExecutionSpace Exec, KokkosView SendV>
auto isend(Communicator<MpiSpace, Exec>& comm, const SendV& sv, int dst, int tag) -> Request<MpiSpace> {
  return isend(comm, sv, dst, tag, DefaultCommMode{});
}
// Search & replace API overload
template <KokkosExecutionSpace Exec, KokkosView SendV>
[[deprecated("Prefer `isend(Communicator, SendV, int, int)` overload")]] auto isend(
    const Exec& exec, const SendV& sv, int dst, int tag, MPI_Comm comm
) -> Request<MpiSpace> {
  auto communicator = Communicator<MpiSpace, Exec>::from_raw(comm, exec);
  return isend(communicator, sv, dst, tag, DefaultCommMode{});
}

template <KokkosExecutionSpace Exec, KokkosView SendV, CommunicationMode SendMode>
auto send(Communicator<MpiSpace, Exec>& comm, const SendV& sv, int dst, int tag, SendMode) -> void {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::send");

  auto mpi_send_fn = [dst, tag](void* buf, int count, MPI_Datatype dtype, MPI_Comm mpi_comm) {
    if constexpr (std::is_same_v<SendMode, CommModeStandard>) {
      MPI_Send(buf, count, dtype, dst, tag, mpi_comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeReady>) {
      MPI_Rsend(buf, count, dtype, dst, tag, mpi_comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeSynchronous>) {
      MPI_Ssend(buf, count, dtype, dst, tag, mpi_comm);
    } else {
      static_assert(std::is_void_v<SendMode>, "KokkosComm::mpi::send: unexpected communication mode");
    }
  };

  const auto exec = comm.exec();
  if (is_contiguous(sv)) {
    exec.fence("fence before send");
    mpi_send_fn(data_handle(sv), span(sv), datatype_for<MpiSpace>(sv), comm.comm());
  } else {
    using Packer = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv = Packer::pack(exec, "pkd_sv", sv);
    exec.fence("fence before send");
    mpi_send_fn(data_handle(pckd_sv.view), pckd_sv.count, pckd_sv.datatype, comm.comm());
  }

  Kokkos::Tools::popRegion();
}
template <KokkosExecutionSpace Exec, KokkosView SendV>
auto send(Communicator<MpiSpace, Exec>& comm, const SendV& sv, int dst, int tag) -> void {
  send(comm, sv, dst, tag, DefaultCommMode{});
}
// Search & replace API overload
template <KokkosExecutionSpace Exec, KokkosView SendV>
[[deprecated("Prefer `send(Communicator, SendV, int, int)` overload")]] auto send(
    const Exec& exec, const SendV& sv, int dst, int tag, MPI_Comm comm
) -> void {
  auto communicator = Communicator<MpiSpace, Exec>::from_raw(comm, exec);
  send(communicator, sv, dst, tag, DefaultCommMode{});
}

}  // namespace KokkosComm::mpi
