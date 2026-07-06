// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/impl/view_preparation.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
auto alltoall(const ExecSpace& space, const SendView& sv, const RecvView& rv, int count, ncclComm_t comm)
    -> Request<NcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::alltoall: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::alltoall");

  Request<NcclSpace> req;
  auto send_ready = KC::Impl::prepare<KC::Impl::ViewAccess::Read>(space, sv, req);
  auto recv_ready = KC::Impl::prepare<KC::Impl::ViewAccess::Write>(space, rv, req);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  ncclAlltoAll(send_ready.buf_ptr(), recv_ready.buf_ptr(), count, send_ready.datatype(), comm, space.cuda_stream());
#else
  int n_pes;
  auto* send_ptr = static_cast<ST*>(send_ready.buf_ptr());
  auto* recv_ptr = static_cast<RT*>(recv_ready.buf_ptr());
  ncclCommCount(comm, &n_pes);
  ncclGroupStart();
  for (int r = 0; r < n_pes; ++r) {
    ncclSend(send_ptr + r * count, count, send_ready.datatype(), r, comm, space.cuda_stream());
    ncclRecv(recv_ptr + r * count, count, recv_ready.datatype(), r, comm, space.cuda_stream());
  }
  ncclGroupEnd();
#endif
  req.capture_stream_state(space.cuda_stream());

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView>
struct AllToAll<SendView, RecvView, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, const SendView sv, RecvView rv, int count)
      -> Request<NcclSpace> {
    return nccl::alltoall(h.exec(), sv, rv, count, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
