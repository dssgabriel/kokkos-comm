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

namespace KokkosComm::Experimental::nccl::Impl {

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
auto recv(const ExecSpace &space, RecvView &rv, int peer, ncclComm_t comm) -> void {
  using RecvScalar = typename RecvView::value_type;

  Kokkos::Tools::pushRegion("KokkosComm::Impl::recv");

  if (!KokkosComm::is_contiguous(rv)) {
    using Packer = typename Impl::PackTraits<RecvView>::packer_type;
    auto args    = Packer::pack(space, rv);
    // TODO: consider using a private stream pool in order to avoid synchronizing the underlying stream (which may not
    // be empty and have in-flight communications we don't want to wait on)
    space.fence();  // make sure allocation is complete before receiving

    ncclRecv(KokkosComm::data_handle(args.view_), KokkosComm::span(args.view_), datatype_v<RecvScalar>, peer, comm,
             space.cuda_stream());
    Packer::unpack_into(space, rv, args.view_);
  } else {
    ncclRecv(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype_v<RecvScalar>, peer, comm,
             space.cuda_stream());
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Experimental::nccl::Impl
