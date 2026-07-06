// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/impl/view_preparation.hpp>
#include "mpi_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/tags.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Impl {

// Recv implementation for Mpi
template <KokkosExecutionSpace ExecSpace, MutKokkosView RecvView>
struct Recv<RecvView, ExecSpace, MpiSpace> {
  static Request<MpiSpace> execute(Communicator<MpiSpace, ExecSpace>& h, const RecvView& rv, int src) {
    const ExecSpace& space = h.exec();

    Request<MpiSpace> req;
    auto ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Write>(space, rv, req);
    // Ensure any view-preparation work on `space` is ordered before MPI writes the receive buffer.
    space.fence("fence before irecv");
    MPI_Irecv(ready.buf_ptr(), ready.count(), ready.datatype(), src, POINTTOPOINT_TAG, h.comm(), req.request_ptr());
    return req;
  }
};

}  // namespace Impl
namespace mpi {

template <MutKokkosView RecvView>
void irecv(const RecvView& rv, int src, int tag, MPI_Comm comm, MPI_Request& req) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::irecv");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "Only contiguous irecv viewsupported");

  using RecvScalar = typename RecvView::non_const_value_type;
  MPI_Irecv(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, RecvScalar>(), src, tag, comm, &req);

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
}  // namespace KokkosComm
