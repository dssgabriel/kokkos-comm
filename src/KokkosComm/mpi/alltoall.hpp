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

template <KokkosExecutionSpace ExecSpace, KokkosView SView, MutKokkosView RView>
auto ialltoall(const ExecSpace& space, const SView sv, RView rv, int count, MPI_Comm comm) -> Request<MpiSpace> {
  using ST = typename SView::non_const_value_type;
  using RT = typename RView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::ialltoall: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::mpi::ialltoall");

  Request<MpiSpace> req;
  auto send_ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Read>(space, sv, req);
  auto recv_ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Write>(space, rv, req);

  // Ensure packing/staging copies enqueued on `space` complete before MPI reads the send buffer.
  space.fence("fence before non-blocking all-to-all");
  // All ranks send/recv same count
  MPI_Ialltoall(
      send_ready.buf_ptr(), count, send_ready.datatype(), recv_ready.buf_ptr(), count, recv_ready.datatype(), comm,
      req.request_ptr()
  );

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
void alltoall(
    const ExecSpace& space,
    const SendView& sv,
    const size_t sendCount,
    const RecvView& rv,
    const size_t recvCount,
    MPI_Comm comm
) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::alltoall");

  int size;
  MPI_Comm_size(comm, &size);

  if (sendCount * size > KokkosComm::extent(sv, 0)) {
    std::stringstream ss;
    ss << "alltoall sendCount * communicator size (" << sendCount << " * " << size
       << ") is greater than send view size";
    KokkosComm::mpi::fail_if(true, ss.str().data());
  }
  if (recvCount * size > KokkosComm::extent(rv, 0)) {
    std::stringstream ss;
    ss << "alltoall recvCount * communicator size (" << recvCount << " * " << size
       << ") is greater than recv view size";
    KokkosComm::mpi::fail_if(true, ss.str().data());
  }

  Request<MpiSpace> req;
  auto send_ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Read>(space, sv, req);
  auto recv_ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Write>(space, rv, req);
  // Ensure packing/staging copies enqueued on `space` complete before MPI reads the send buffer.
  space.fence("fence before alltoall");
  MPI_Ialltoall(
      send_ready.buf_ptr(), sendCount, send_ready.datatype(), recv_ready.buf_ptr(), recvCount, recv_ready.datatype(),
      comm, req.request_ptr()
  );
  req.wait();

  Kokkos::Tools::popRegion();
}

// in-place alltoall
template <KokkosExecutionSpace ExecSpace, MutKokkosView RecvView>
void alltoall(const ExecSpace& space, const RecvView& rv, const size_t recvCount, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::alltoall");

  int size;
  MPI_Comm_size(comm, &size);

  if (recvCount * size > KokkosComm::extent(rv, 0)) {
    std::stringstream ss;
    ss << "alltoall recvCount * communicator size (" << recvCount << " * " << size
       << ") is greater than recv view size";
    KokkosComm::mpi::fail_if(true, ss.str().data());
  }

  Request<MpiSpace> req;
  auto ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::ReadWrite>(space, rv, req);
  // Ensure packing/staging copies enqueued on `space` complete before MPI reads or writes the buffer.
  space.fence("fence before alltoall");
  MPI_Ialltoall(
      MPI_IN_PLACE, 0 /*ignored*/, MPI_BYTE /*ignored*/, ready.buf_ptr(), recvCount, ready.datatype(), comm,
      req.request_ptr()
  );
  req.wait();

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
namespace Experimental::Impl {

template <KokkosView SendView, MutKokkosView RecvView, KokkosExecutionSpace ExecSpace>
struct AllToAll<SendView, RecvView, ExecSpace, MpiSpace> {
  static auto execute(Communicator<MpiSpace, ExecSpace>& h, const SendView sv, RecvView rv, int count)
      -> Request<MpiSpace> {
    return mpi::ialltoall(h.exec(), sv, rv, count, h.comm());
  }
};

}  // namespace Experimental::Impl
}  // namespace KokkosComm
