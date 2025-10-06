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

#include "impl/types.hpp"
#include "impl/pack_traits.hpp"

namespace KokkosComm {
namespace Experimental::nccl {

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
auto recv(const ExecSpace &space, RecvView &rv, int peer, ncclComm_t comm) -> Req<Nccl> {
  using T = typename RecvView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::recv");

  Req<Nccl> req{space.cuda_stream()};
  if (is_contiguous(rv)) {
    ncclRecv(data_handle(rv), span(rv), Impl::datatype_v<T>, peer, comm, space.cuda_stream());
  } else {
    using Packer = typename Impl::PackTraits<T>::packer_type;
    auto args    = Packer::pack(space, rv);
    // TODO: Consider using a private stream pool to avoid synchronizing the underlying stream, which may not
    // be empty and have in-flight communications we do not want to wait on.
    space.fence();  // make sure allocation is complete before receiving

    ncclRecv(data_handle(args.view_), span(args.view_), Impl::datatype_v<T>, peer, comm, space.cuda_stream());
    Packer::unpack_into(space, rv, args.view_);
    req.extend_view_lifetime(args.view_);
  }
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::nccl
namespace Impl {

template <KokkosView RecvView>
struct Recv<Kokkos::Cuda, Experimental::Nccl> {
  static auto execute(Handle<Kokkos::Cuda, Experimental::Nccl> &h, RecvView sv, int peer) -> Req<Experimental::Nccl> {
    return Experimental::nccl::recv(h.space(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
