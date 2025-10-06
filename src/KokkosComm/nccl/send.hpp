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

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
auto send(const ExecSpace& space, const SendView& sv, int peer, ncclComm_t comm) -> Req<Nccl> {
  using T = typename SendView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  Req<Nccl> req{space.cuda_stream()};
  if (is_contiguous(sv)) {
    ncclSend(data_handle(sv), span(sv), Impl::datatype_v<T>, peer, comm, space.cuda_stream());
  } else {
    using Packer = typename Impl::PackTraits<SendView>::packer_type;
    auto args    = Packer::pack(space(), sv);
    // TODO: Consider using a private stream pool to avoid synchronizing the underlying stream, which may not
    // be empty and have in-flight communications we do not want to wait on.
    space().fence();

    ncclSend(data_handle(args.view_), span(args.view_), Impl::datatype_v<T>, peer, comm, space.cuda_stream());
    Packer::unpack_into(space, sv, args.view_);
    req.extend_view_lifetime(args.view_);
  }
  req.extend_view_lifetime(sv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::nccl
namespace Impl {

template <KokkosView SendView>
struct Send<Kokkos::Cuda, Experimental::Nccl> {
  static auto execute(Handle<Kokkos::Cuda, Experimental::Nccl>& h, SendView sv, int peer) -> Req<Experimental::Nccl> {
    return Experimental::nccl::send(h.space(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
