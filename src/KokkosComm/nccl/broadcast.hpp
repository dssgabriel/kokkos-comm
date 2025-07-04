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

namespace KokkosComm::Experimental::nccl::Impl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto broadcast(const ExecSpace &space, const SendView &sv, const RecvView &rv, int root, ncclComm_t comm) -> void {
  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;
  static_assert(std::is_same_v<SendScalar, RecvScalar>, "nccl::broadcast: View value types must be identical");
  static_assert(KokkosComm::rank<SendView>() <= 1 && KokkosComm::rank<RecvView>() <= 1,
                "nccl::broadcast: only rank-1 Views are supported");

  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::Impl::broadcast");

  if (!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv)) {
    Kokkos::abort("nccl::broadcast: unimplemented for non-contiguous views");
  } else {
    ncclBroadcast(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), KokkosComm::span(sv),
                  datatype_v<SendScalar>, root, comm, space.cuda_stream());
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Experimental::nccl::Impl
