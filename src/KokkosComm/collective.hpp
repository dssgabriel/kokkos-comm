// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <utility>

#include <Kokkos_Core_fwd.hpp>

#include <KokkosComm/fwd.hpp>
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <KokkosComm/nccl/nccl_space.hpp>
#include <KokkosComm/nccl/broadcast.hpp>
#include <KokkosComm/nccl/allgather.hpp>
#include <KokkosComm/nccl/allreduce.hpp>
#include <KokkosComm/nccl/reduce.hpp>
#endif

namespace KokkosComm::Experimental {

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto reduce(Handle<ExecSpace, CommSpace>& h, const SendView& sv, RecvView& rv, int root) -> Req<CommSpace> {
  return Impl::Reduce<SendView, RecvView, RedOp, ExecSpace, CommSpace>::execute(h, sv, rv, root);
}

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto reduce(const SendView& sv, RecvView& rv, int root) -> Req<CommSpace> {
  return reduce(Handle<ExecSpace, CommSpace>{}, sv, rv, root);
}

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto allgather(Handle<ExecSpace, CommSpace>& h, const SendView& sv, RecvView& rv) -> Req<CommSpace> {
  return Impl::AllGather<SendView, RecvView, ExecSpace, CommSpace>::execute(h, sv, rv);
}

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto allgather(const SendView& sv, RecvView& rv) -> Req<CommSpace> {
  return allgather(Handle<ExecSpace, CommSpace>{}, sv, rv);
}

}  // namespace KokkosComm::Experimental
