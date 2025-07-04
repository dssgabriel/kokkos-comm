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

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
auto send(const ExecSpace& space, const SendView& sv, int peer, ncclComm_t comm) -> void {
  using SendScalar = typename SendView::value_type;

  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  if (!KokkosComm::is_contiguous(sv)) {
    using Packer = typename Impl::PackTraits<SendView>::packer_type;
    auto args    = Packer::pack(space(), sv);
    // TODO: consider using a private stream pool in order to avoid synchronizing the underlying stream (which may not
    // be empty and have in-flight communications we don't want to wait on)
    space().fence();

    ncclSend(KokkosComm::data_handle(args.view_), KokkosComm::span(args.view_), datatype_v<SendScalar>, peer, comm,
             space.cuda_stream());
    Packer::unpack_into(space, sv, args.view_);
  } else {
    ncclSend(KokkosComm::data_handle(sv), KokkosComm::span(sv), datatype_v<SendScalar>, peer, comm,
             space.cuda_stream());
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Experimental::nccl::Impl
