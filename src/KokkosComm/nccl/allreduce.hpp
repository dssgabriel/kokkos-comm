// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/reduction_op.hpp>
#include <KokkosComm/impl/view_preparation.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
auto allreduce(const ExecSpace& space, const SendView& sv, const RecvView& rv, ncclRedOp_t op, ncclComm_t comm)
    -> Request<NcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(
      std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::allreduce: View value types must be identical"
  );
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::allreduce");

  Request<NcclSpace> req;
  auto send_ready = KC::Impl::prepare<KC::Impl::ViewAccess::Read>(space, sv, req);
  auto recv_ready = KC::Impl::prepare<KC::Impl::ViewAccess::Write>(space, rv, req);
  ncclAllReduce(
      send_ready.buf_ptr(), recv_ready.buf_ptr(), send_ready.count(), send_ready.datatype(), op, comm,
      space.cuda_stream()
  );
  req.capture_stream_state(space.cuda_stream());

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView, ReductionOperator RedOp>
struct AllReduce<SendView, RecvView, RedOp, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, const SendView sv, RecvView rv) -> Request<NcclSpace> {
    return nccl::allreduce(h.exec(), sv, rv, reduction_op<NcclSpace, RedOp>(), h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
