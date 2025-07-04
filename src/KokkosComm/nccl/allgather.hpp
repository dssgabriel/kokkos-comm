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
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::Impl::allgather");

auto allgather(const ExecSpace &space, const SendView &sv, const RecvView &rv, ncclComm_t comm) -> void {
  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;

  static_assert(std::is_same_v<SendScalar, RecvScalar>, "NCCL allgather requires View value types to be identical");
  static_assert(KokkosComm::rank<SendView>() <= 1, "allgather for SendView::rank > 1 not supported");
  static_assert(KokkosComm::rank<RecvView>() <= 1, "allgather for RecvView::rank > 1 not supported");

  if (!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv)) {
    using SPT = PackTraits<SendView>;
    using RPT = PackTraits<RecvView>;

    throw std::runtime_error("allgather for non-contiguous views not implemented");
  } else {
    constexpr auto count = KokkosComm::span(sv);  // all ranks recv `nranks * count`
    ncclAllGather(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), count, datatype_v<SendScalar>, comm,
                  space.cuda_stream());
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Experimental::nccl::Impl
