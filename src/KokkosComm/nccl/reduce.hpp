// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <nccl.h>
#include <Kokkos_Core.hpp>

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

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
auto reduce(
    const ExecSpace& space, const SendView& sv, RecvView& rv, ncclRedOp_t op, int root, int /*rank*/, ncclComm_t comm
) -> Request<NcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::reduce: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::reduce");

  Request<NcclSpace> req;
  auto send_ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Read>(space, sv, req);
  auto recv_ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Write>(space, rv, req);
  ncclReduce(
      send_ready.buf_ptr(), recv_ready.buf_ptr(), send_ready.count(), send_ready.datatype(), op, root, comm,
      space.cuda_stream()
  );
  req.capture_stream_state(space.cuda_stream());

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView, ReductionOperator RedOp>
struct Reduce<SendView, RecvView, RedOp, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, const SendView sv, RecvView rv, int root)
      -> Request<NcclSpace> {
    return nccl::reduce(h.exec(), sv, rv, reduction_op<NcclSpace, RedOp>(), root, h.rank(), h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
