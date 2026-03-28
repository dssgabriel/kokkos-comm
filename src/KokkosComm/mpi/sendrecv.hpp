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

#include "impl/pack_traits.hpp"
#include "impl/tags.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace Exec, KokkosView SendV, KokkosView RecvV>
auto isendrecv(
    Communicator<MpiSpace, Exec>& comm, const SendV& sv, int dst, int stag, const RecvV& rv, int src, int rtag
) -> Request<MpiSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::isendrecv");

  const auto exec = comm.exec();
  Request<MpiSpace> req;

  // TODO: Refactor once PackTraits returns packing strategy refactor lands.
  // Send/Recv sides are independent and this 4-way branch disappears.
  if (is_contiguous(sv) and is_contiguous(rv)) {
    exec.fence("fence before MPI_Isendrecv");
    MPI_Isendrecv(
        data_handle(sv), span(sv), datatype_for<MpiSpace>(sv), dst, stag, data_handle(rv), span(rv),
        datatype_for<MpiSpace>(rv), src, rtag, comm.comm(), req.request_ptr()
    );
  } else if (!is_contiguous(sv) and is_contiguous(rv)) {
    using SendPckr = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv   = SendPckr::pack(exec, "pckd_sv", sv);
    exec.fence("fence sv packing before MPI_Isendrecv");
    MPI_Isendrecv(
        data_handle(pckd_sv.view), pckd_sv.count, pckd_sv.datatype, dst, stag, data_handle(rv), span(rv),
        datatype_for<MpiSpace>(rv), src, rtag, comm.comm(), req.request_ptr()
    );
    req.extend_view_lifetime(pckd_sv.view);
  } else if (is_contiguous(sv) and !is_contiguous(rv)) {
    using RecvPckr = typename Impl::PackTraits<RecvV>::packer_type;
    auto pckd_rv   = RecvPckr::allocate_packed_for(exec, "pckd_rv", rv);
    exec.fence("fence rv packed allocation before MPI_Isendrecv");
    MPI_Isendrecv(
        data_handle(sv), span(sv), datatype_for<MpiSpace>(sv), dst, stag, data_handle(pckd_rv.view), pckd_rv.count,
        pckd_rv.datatype, src, rtag, comm.comm(), req.request_ptr()
    );
    // Implicitly extends `pckd_rv.view` and `rv` lifetime due to lambda capture
    req.add_callback([exec, rv, pckd_rv]() {
      RecvPckr::unpack_into(exec, rv, pckd_rv.view);
      exec.fence("fence unpacking after MPI_Isendrecv");
    });
  } else if (!is_contiguous(sv) and !is_contiguous(rv)) {
    using SendPckr = typename Impl::PackTraits<SendV>::packer_type;
    using RecvPckr = typename Impl::PackTraits<RecvV>::packer_type;
    auto pckd_sv   = SendPckr::pack(exec, "pckd_sv", sv);
    auto pckd_rv   = RecvPckr::allocate_packed_for(exec, "pckd_rv", rv);
    exec.fence("fence sv packing and rv packed allocation before MPI_Isendrecv");
    MPI_Isendrecv(
        data_handle(pckd_sv.view), pckd_sv.count, pckd_sv.datatype, dst, stag, data_handle(pckd_rv.view), pckd_rv.count,
        pckd_rv.datatype, src, rtag, comm.comm(), req.request_ptr()
    );
    req.extend_view_lifetime(pckd_sv.view);
    // Implicitly extends `pckd_rv.view` and `rv` lifetime due to lambda capture
    req.add_callback([exec, rv, pckd_rv]() {
      RecvPckr::unpack_into(exec, rv, pckd_rv.view);
      exec.fence("fence unpacking after MPI_Isendrecv");
    });
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace KokkosComm::mpi
