// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <utility>

#include <Kokkos_Core_fwd.hpp>

#include "fwd.hpp"
#include "reduction_op.hpp"
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/nccl_space.hpp"
#include "nccl/handle.hpp"
#include "nccl/req.hpp"
#include "nccl/broadcast.hpp"
#include "nccl/alltoall.hpp"
#include "nccl/allgather.hpp"
#include "nccl/allreduce.hpp"
#include "nccl/reduce.hpp"
#endif

namespace KokkosComm::Experimental {

/// Broadcast w/ explicit handle
template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto broadcast(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv, int root) -> Req<CommSpace> {
  return Impl::Broadcast<SendView, RecvView, ExecSpace, CommSpace>::execute(h, sv, rv, root);
}

/// In-place broadcast w/ explicit handle
template <KokkosView View, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto broadcast(Handle<ExecSpace, CommSpace>& h, View v, int root) -> Req<CommSpace> {
  return Impl::Broadcast<View, View, ExecSpace, CommSpace>::execute(h, v, v, root);
}

/// All-gather w/ explicit handle
template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto allgather(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv) -> Req<CommSpace> {
  return Impl::AllGather<SendView, RecvView, ExecSpace, CommSpace>::execute(h, sv, rv);
}

/// All-to-all w/ explicit handle
template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto alltoall(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv, int count) -> Req<CommSpace> {
  return Impl::AllToAll<SendView, RecvView, ExecSpace, CommSpace>::execute(h, sv, rv, count);
}

/// All-reduce w/ explicit handle
template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto allreduce(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv, RedOp) -> Req<CommSpace> {
  return Impl::AllReduce<SendView, RecvView, RedOp, ExecSpace, CommSpace>::execute(h, sv, rv);
}

/// Reduce w/ explicit handle
template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto reduce(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv, int root, RedOp) -> Req<CommSpace> {
  return Impl::Reduce<SendView, RecvView, RedOp, ExecSpace, CommSpace>::execute(h, sv, rv, root);
}

}  // namespace KokkosComm::Experimental
