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

template <KokkosExecutionSpace Ex, KokkosView SendV, CommunicationMode SendMode>
auto isend(const Ex& exec, const SendV& sv, int dst, int tag, MPI_Comm comm, SendMode) -> Request<MpiSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::isend");

  auto mpi_isend_fn = [dst, tag, comm](void* buf, int count, MPI_Datatype dtype, MPI_Request* req_ptr) {
    if constexpr (std::is_same_v<SendMode, mpi::CommModeStandard>) {
      MPI_Isend(buf, count, dtype, dst, tag, comm, req_ptr);
    } else if constexpr (std::is_same_v<SendMode, mpi::CommModeReady>) {
      MPI_Irsend(buf, count, dtype, dst, tag, comm, req_ptr);
    } else if constexpr (std::is_same_v<SendMode, mpi::CommModeSynchronous>) {
      MPI_Issend(buf, count, dtype, dst, tag, comm, req_ptr);
    } else {
      static_assert(std::is_void_v<SendMode>, "unexpected communication mode");
    }
  };

  Request<MpiSpace> req;
  if (is_contiguous(sv)) {
    exec.fence("fence before MPI_Isend");
    mpi_isend_fn(data_handle(sv), span(sv), datatype_for<MpiSpace>(sv), dst, tag, comm, req.request_ptr());
  } else {
    using Packer = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv = Packer::pack(exec, "pckd_sv", sv);
    exec.fence("fence packed allocation before MPI_Isend");
    mpi_isend_fn(data_handle(pckd_sv.view), pckd_sv.count, pckd_sv.datatype, dst, tag, comm, req.request_ptr());
    req.extend_view_lifetime(pckd_sv.view);
  }
  req.extend_view_lifetime(sv);

  Kokkos::Tools::popRegion();
  return req;
}
template <KokkosExecutionSpace Ex, KokkosView SendV>
auto isend(const Ex& exec, const SendV& sv, int dst, int tag, MPI_Comm comm) -> void {
  isend(exec, sv, dst, tag, comm, DefaultCommMode{});
}

template <KokkosExecutionSpace Ex, KokkosView SendV, CommunicationMode SendMode>
auto send(const Ex& exec, const SendV& sv, int dst, int tag, MPI_Comm comm, SendMode) -> void {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::send");

  auto mpi_send_fn = [dst, tag, comm](void* buf, int count, MPI_Datatype dtype) {
    if constexpr (std::is_same_v<SendMode, CommModeStandard>) {
      MPI_Send(buf, count, dtype, dst, tag, comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeReady>) {
      MPI_Rsend(buf, count, dtype, dst, tag, comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeSynchronous>) {
      MPI_Ssend(buf, count, dtype, dst, tag, comm);
    } else {
      static_assert(std::is_void_v<SendMode>, "KokkosComm::mpi::send: unexpected communication mode");
    }
  };

  if (is_contiguous(sv)) {
    exec.fence("fence before send");
    mpi_send_fn(data_handle(sv), span(sv), datatype_for<MpiSpace>(sv));
  } else {
    using Packer = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv = Packer::pack(exec, "pkd_sv", sv);
    exec.fence("fence before send");
    mpi_send_fn(data_handle(pckd_sv.view), pckd_sv.count, pckd_sv.datatype);
  }

  Kokkos::Tools::popRegion();
}
template <KokkosExecutionSpace Ex, KokkosView SendV>
auto send(const Ex& exec, const SendV& sv, int dst, int tag, MPI_Comm comm) -> void {
  send(exec, sv, dst, tag, comm, DefaultCommMode{});
}

}  // namespace KokkosComm::mpi
