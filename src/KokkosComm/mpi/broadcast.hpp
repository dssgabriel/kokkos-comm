// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/impl/view_preparation.hpp>
#include "mpi_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace mpi {

template <KokkosExecutionSpace ExecSpace, MutKokkosView View>
auto ibroadcast(const ExecSpace& space, View& v, int root, MPI_Comm comm) -> Request<MpiSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::ibroadcast");

  Request<MpiSpace> req;
  auto ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::ReadWrite>(space, v, req);
  // Ensure packing/staging copies enqueued on `space` complete before MPI reads or writes the buffer.
  space.fence("fence before non-blocking broadcast");
  MPI_Ibcast(ready.buf_ptr(), ready.count(), ready.datatype(), root, comm, req.request_ptr());

  Kokkos::Tools::popRegion();
  return req;
}

template <MutKokkosView View>
void broadcast(View const& v, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::broadcast");

  using Scalar = typename View::value_type;

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(v), "low-level broadcast requires contiguous view");

  MPI_Bcast(KokkosComm::data_handle(v), KokkosComm::span(v), datatype<MpiSpace, Scalar>(), root, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, MutKokkosView View>
void broadcast(ExecSpace const& space, View const& v, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::broadcast");

  ibroadcast(space, v, root, comm).wait();

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
namespace Experimental::Impl {

template <MutKokkosView View, KokkosExecutionSpace ExecSpace>
struct Broadcast<View, ExecSpace, MpiSpace> {
  static auto execute(Communicator<MpiSpace, ExecSpace>& h, View& v, int root) -> Request<MpiSpace> {
    return KokkosComm::mpi::ibroadcast(h.exec(), v, root, h.comm());
  }
};

}  // namespace Experimental::Impl
}  // namespace KokkosComm
