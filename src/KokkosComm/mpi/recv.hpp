// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/impl/view_preparation.hpp>

#include "request.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <MutKokkosView RecvView>
void recv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Status *status) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "only contiguous views supported for low-level recv");

  using ScalarType = typename RecvView::non_const_value_type;
  MPI_Recv(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, ScalarType>(), src, tag, comm, status);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, MutKokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");

  Request<MpiSpace> req;
  auto ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Write>(space, rv, req);
  // Ensure any view-preparation work on `space` is ordered before MPI writes the receive buffer.
  space.fence("fence before recv");
  MPI_Irecv(ready.buf_ptr(), ready.count(), ready.datatype(), src, tag, comm, req.request_ptr());
  req.wait();

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
