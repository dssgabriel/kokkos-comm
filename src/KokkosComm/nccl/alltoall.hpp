//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

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
auto alltoall(const ExecSpace &space, const SendView &sv, const RecvView &rv, int count, ncclComm_t comm) -> Req<Nccl> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::alltoall: View value types must be identical");
  static_assert(rank<SendView>() == 1 and rank<RecvView>() == 1,
                "KokkosComm::Experimental::nccl::alltoall: only rank-1 Views are supported");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::alltoall");

  Req<Nccl> req{space.cuda_stream()};
  if (is_contiguous(sv) and is_contiguous(rv)) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    ncclAlltoAll(data_handle(sv), data_handle(rv), count, Impl::datatype_v<ST>, comm, space.cuda_stream());
#else
    int n_pes;
    ncclCommCount(comm, &n_pes);
    ncclGroupStart();
    for (int r = 0; r < n_pres; ++r) {
      ncclSend(data_handle(sv) + r * count, count, Impl::datatype_v<ST>, r, comm, space.cuda_stream());
      ncclRecv(data_handle(rv) + r * count, count, Impl::datatype_v<RT>, r, comm, space.cuda_stream());
    }
    ncclGroupEnd();
#endif
  } else {
    Kokkos::abort("KokkosComm::Experimental::nccl::alltoall: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

struct AllToAll<Kokkos::Cuda, Nccl> {
  static auto execute(Handle<Kokkos::Cuda, Nccl> &h, const SendView sv, RecvView rv, int count) -> Req<Nccl> {
    return nccl::alltoall(h.space(), sv, rv, count, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
