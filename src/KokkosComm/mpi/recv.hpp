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

template <KokkosExecutionSpace Exec, KokkosView RecvV>

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "only contiguous views supported for low-level recv");

  using ScalarType = typename RecvView::non_const_value_type;
  MPI_Recv(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, ScalarType>(), src, tag, comm, status);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace Exec, KokkosView RecvV>
auto recv(Communicator<MpiSpace, Exec>& comm, const RecvV& rv, int src, int tag, MPI_Status* status) -> void {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");

  const auto exec = comm.exec();
  if (is_contiguous(rv)) {
    exec.fence("fence before MPI_Recv");
    MPI_Recv(data_handle(rv), span(rv), datatype_for<MpiSpace>(rv), src, tag, comm.comm(), status);
  } else {
    using Packer = typename Impl::PackTraits<RecvV>::packer_type;
    auto pckd_rv = Packer::allocate_packed_for(exec, "pckd_rv", rv);
    exec.fence("fence packed allocation before MPI_Recv");
    MPI_Recv(data_handle(pckd_rv.view), pckd_rv.count, pckd_rv.datatype, src, tag, comm.comm(), status);
    Packer::unpack_into(exec, rv, pckd_rv.view);
    exec.fence("fence unpacking after MPI_Recv");
  }

  Kokkos::Tools::popRegion();
}
template <KokkosExecutionSpace Exec, KokkosView RecvV>
auto recv(Communicator<MpiSpace, Exec>& comm, const RecvV& rv, int src, int tag) -> void {
  recv(comm, rv, src, tag, MPI_STATUS_IGNORE);
}
// Search & replace API overloads
template <KokkosExecutionSpace Exec, KokkosView RecvV>
[[deprecated("Prefer `recv(Communicator, RecvV, int, int, MPI_Status)` overload")]] auto recv(
    const Exec& exec, const RecvV& rv, int src, int tag, MPI_Comm comm, MPI_Status* status
) -> void {
  auto communicator = Communicator<MpiSpace, Exec>::from_raw(comm, exec);
  recv(communicator, rv, src, tag, status);
}
template <KokkosExecutionSpace Exec, KokkosView RecvV>
[[deprecated("Prefer `recv(Communicator, RecvV, int, int)` overload")]] auto recv(
    const Exec& exec, const RecvV& rv, int src, int tag, MPI_Comm comm
) -> void {
  auto communicator = Communicator<MpiSpace, Exec>::from_raw(comm, exec);
  recv(communicator, rv, src, tag, MPI_STATUS_IGNORE);
}

}  // namespace KokkosComm::mpi
