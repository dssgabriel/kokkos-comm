// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/pack_traits.hpp"
#include "impl/types.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto allreduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, ncclRedOp_t op, ncclComm_t comm)
    -> Req<Nccl> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>,
                "KokkosComm::Experimental::nccl::allreduce: View value types must be identical");
  static_assert(rank<SendView>() == 1 and rank<RecvView>() == 1,
                "KokkosComm::Experimental::nccl::allreduce: only rank-1 Views are supported");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::allreduce");

  Req<Nccl> req{space.cuda_stream()};
  if (is_contiguous(sv) and is_contiguous(rv)) {
    ncclAllReduce(data_handle(sv), data_handle(rv), span(sv), Impl::datatype_v<ST>, op, comm, space.cuda_stream());
  } else {
    Kokkos::abort("KokkosComm::Experimental::nccl::allreduce: unimplemented for non-contiguous Views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp>
struct AllReduce<SendView, RecvView, RedOp, Kokkos::Cuda, Nccl> {
  static auto execute(Handle<Kokkos::Cuda, Nccl> &h, const SendView sv, RecvView rv) -> Req<Nccl> {
    return nccl::allreduce(h.space(), sv, rv, nccl::Impl::reduction_op_v<RedOp>, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
