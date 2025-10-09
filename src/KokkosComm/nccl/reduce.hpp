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
auto reduce(const ExecSpace &space, const SendView sv, RecvView rv, ncclRedOp_t op, int root, int rank, ncclComm_t comm)
    -> Req<Nccl> {
  using SendPacker = typename PackTraits<SendView>::packer_type;
  using RecvPacker = typename PackTraits<RecvView>::packer_type;
  using ST         = typename SendView::non_const_value_type;
  using RT         = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::reduce: View value types must be identical");
  static_assert(rank<SendView>() == 1 and rank<RecvView>() == 1,
                "KokkosComm::Experimental::nccl::reduce: only rank-1 Views are supported");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::reduce");

  Req<Nccl> req{space.cuda_stream()};
  if (is_contiguous(sv)) {
    if (rank != root and is_contiguous(rv)) {
      ncclReduce(data_handle(sv), data_handle(rv), span(sv), datatype_v<ST>, op, root, comm, space.cuda_stream());
    } else {
      auto recv_args = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      ncclReduce(data_handle(sv), data_handle(recv_args.view_), data_handle(sv), datatype_v<ST>, op, root, comm,
                 space.cuda_stream());
      RecvPacker::unpack_into(space, rv, recv_args.view_);
      req.extend_view_lifetime(recv_args.view_);
    }
  } else {
    auto send_args = SendPacker::pack(space, sv);
    space.fence();
    if (rank != root and is_contiguous(rv)) {
      ncclReduce(data_handle(send_args.view_), data_handle(rv), span(send_args.view_), datatype_v<ST>, op, root, comm,
                 space.cuda_stream());
    } else {
      auto recv_args = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      ncclReduce(data_handle(send_args.view_), data_handle(recv_args.view_), span(send_args.view_), datatype_v<ST>, op,
                 root, comm, space.cuda_stream());
      RecvPacker::unpack_into(space, rv, recv_args.view);
      req.extend_view_lifetime(recv_args.view_);
    }
    req.extend_view_lifetime(send_args.view_);
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp>
struct Reduce<Kokkos::Cuda, Nccl> {
  static auto execute(Handle<Kokkos::Cuda, Nccl> &h, const SendView sv, RecvView rv, int root) -> Req<Nccl> {
    return nccl::reduce(h.space(), sv, rv, nccl::Impl::reduction_op_v<RedOp>, root, h.rank(), h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
