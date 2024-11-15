// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>
#include <KokkosComm/nccl/reduce.hpp>

#include <Kokkos_Core.hpp>

#include <utility>
#include "KokkosComm/nccl/allgather.hpp"
#include "KokkosComm/nccl/nccl.hpp"

namespace KokkosComm::Experimental {

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::Cuda,
          CommunicationSpace CommSpace = KokkosComm::Experimental::Nccl, KokkosComm::ReductionOp RedOp>
auto reduce(const Handle<ExecSpace, CommSpace>& h, const SendView& sv, RecvView& rv, int root) -> Req<Nccl> {
  nccl::Impl::reduce(h.space(), sv, rv, nccl::Impl::reduction_op_v<RedOp>, root, h.rank(), h.get_inner());
  return Req<Nccl>(h.space().cuda_stream());
}

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::Cuda,
          CommunicationSpace CommSpace = KokkosComm::Experimental::Nccl>
auto allgather(const Handle<ExecSpace, CommSpace>& h, const SendView& sv, RecvView& rv) -> Req<Nccl> {
  nccl::Impl::allgather(h.space(), sv, rv, h.get_inner());
  return Req<Nccl>(h.space().cuda_stream());
}

}  // namespace KokkosComm::Experimental
