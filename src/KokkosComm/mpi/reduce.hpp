// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/reduction_op.hpp>
#include <KokkosComm/impl/view_preparation.hpp>
#include "mpi_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SView, MutKokkosView RView>
auto ireduce(const ExecSpace& space, const SView& sv, RView& rv, MPI_Op op, int root, MPI_Comm comm)
    -> Request<MpiSpace> {
// FIXME_EXTERNAL #215
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  // Unsupported if running Open MPI and Views are in CUDA or HIP execution spaces
  static_assert(
#if defined(KOKKOS_ENABLE_CUDA)
      not std::is_same_v<typename SView::execution_space, Kokkos::Cuda> and
          not std::is_same_v<typename RView::execution_space, Kokkos::Cuda>,
#elif defined(KOKKOS_ENABLE_HIP)
      not std::is_same_v<typename SView::execution_space, Kokkos::HIP> and
          not std::is_same_v<typename RView::execution_space, Kokkos::HIP>,
#endif
      "KokkosComm::mpi::ireduce: Unsupported with Open MPI + Kokkos CUDA/HIP backend"
  );
#endif
  using ST = typename SView::non_const_value_type;
  using RT = typename RView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::ireduce: View value types must be identical");
  static_assert(
      rank<SView>() <= 1 and rank<RView>() <= 1,
      "KokkosComm::mpi::ireduce: Views with rank higher than 1 are not supported"
  );
  Kokkos::Tools::pushRegion("KokkosComm::mpi::ireduce");

  const int rank = [=]() {
    int r;
    MPI_Comm_rank(comm, &r);
    return r;
  }();

  Request<MpiSpace> req;
  auto send_ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Read>(space, sv, req);
  if (rank == root) {
    auto recv_ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Write>(space, rv, req);

    // Ensure packing/staging copies enqueued on `space` complete before MPI reads the send buffer.
    space.fence("fence before MPI call");
    MPI_Ireduce(
        send_ready.buf_ptr(), recv_ready.buf_ptr(), send_ready.count(), send_ready.datatype(), op, root, comm,
        req.request_ptr()
    );
  } else {
    // Ensure packing/staging copies enqueued on `space` complete before MPI reads the send buffer.
    space.fence("fence before MPI call");
    MPI_Ireduce(
        send_ready.buf_ptr(), data_handle(rv), send_ready.count(), send_ready.datatype(), op, root, comm,
        req.request_ptr()
    );
  }

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosView SendView, MutKokkosView RecvView>
void reduce(const SendView& sv, RecvView& rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::reduce");

  KokkosComm::mpi::fail_if(
      !KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv),
      "only contiguous views supported for low-level reduce"
  );

  using SendScalar = typename SendView::non_const_value_type;
  MPI_Reduce(
      KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), KokkosComm::span(sv), datatype<MpiSpace, SendScalar>(),
      op, root, comm
  );

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
void reduce(const ExecSpace& space, const SendView& sv, RecvView& rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::reduce");

  ireduce(space, sv, rv, op, root, comm).wait();

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
namespace Experimental::Impl {

template <KokkosView SendView, MutKokkosView RecvView, ReductionOperator RedOp, KokkosExecutionSpace ExecSpace>
struct Reduce<SendView, RecvView, RedOp, ExecSpace, MpiSpace> {
  static auto execute(Communicator<MpiSpace, ExecSpace>& h, const SendView sv, RecvView rv, int root)
      -> Request<MpiSpace> {
    return mpi::ireduce(h.exec(), sv, rv, reduction_op<MpiSpace, RedOp>(), root, h.comm());
  }
};

}  // namespace Experimental::Impl
}  // namespace KokkosComm
