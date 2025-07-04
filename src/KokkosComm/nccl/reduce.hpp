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
auto reduce(const ExecSpace &space, const SendView sv, RecvView rv, ncclRedOp_t op, int root, int rank, ncclComm_t comm)
    -> void {
  using SendPacker = typename PackTraits<SendView>::packer_type;
  using RecvPacker = typename PackTraits<RecvView>::packer_type;
  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;
  static_assert(std::is_same_v<SendScalar, RecvScalar>, "nccl::reduce: View value types must be identical");
  static_assert(KokkosComm::rank<SendView>() <= 1 && KokkosComm::rank<RecvView>() <= 1,
                "nccl::reduce: only rank-1 Views are supported");

  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::Impl::reduce");

  if (!KokkosComm::is_contiguous(sv)) {
    auto send_args = SendPacker::pack(space, sv);
    space.fence();
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recv_args = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      ncclReduce(KokkosComm::data_handle(send_args.view_), KokkosComm::data_handle(recv_args.view_),
                 KokkosComm::span(send_args.view_), datatype_v<SendScalar>, op, root, comm, space.cuda_stream());
      RecvPacker::unpack_into(space, rv, recv_args.view);
    } else {
      ncclReduce(KokkosComm::data_handle(send_args.view_), KokkosComm::data_handle(rv),
                 KokkosComm::span(send_args.view_), datatype_v<SendScalar>, op, root, comm, space.cuda_stream());
    }
  } else {
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recv_args = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      ncclReduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(recv_args.view_), KokkosComm::data_handle(sv),
                 datatype_v<SendScalar>, op, root, comm, space.cuda_stream());
      RecvPacker::unpack_into(space, rv, recv_args.view);
    } else {
      ncclReduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), KokkosComm::span(sv), datatype_v<SendScalar>,
                 op, root, comm, space.cuda_stream());
    }
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Experimental::nccl::Impl
