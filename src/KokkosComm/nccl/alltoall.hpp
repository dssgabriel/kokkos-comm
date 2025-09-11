// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/pack_traits.hpp"
#include "impl/types.hpp"

namespace KokkosComm::Experimental::nccl::Impl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto alltoall(const ExecSpace &space, const SendView &sv, const RecvView &rv, ncclComm_t comm, int count) -> void {
  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;
  static_assert(std::is_same_v<SendScalar, RecvScalar>, "nccl::alltoall: View value types must be identical");
  static_assert(KokkosComm::rank<SendView>() <= 1 && KokkosComm::rank<RecvView>() <= 1,
                "nccl::alltoall: only rank-1 Views are supported");

  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::Impl::alltoall");

  if (!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv)) {
    Kokkos::abort("nccl::alltoall: unimplemented for non-contiguous views");
  } else {
    ncclAlltoAll(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), count, datatype_v<SendScalar>, comm,
                 space.cuda_stream());
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Experimental::nccl::Impl
