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

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <KokkosComm/nccl/handle.hpp>
#include <KokkosComm/nccl/req.hpp>
#include <KokkosComm/nccl/recv.hpp>
#endif

namespace KokkosComm::Impl {

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <KokkosView RecvView>
struct Recv<Kokkos::Cuda, Kokkos::Experimental::Nccl> {
  static auto execute(Handle<Kokkos::Cuda, KokkosComm::Experimental::Nccl> &h, RecvView sv, int peer)
      -> Req<KokkosComm::Experimental::Nccl> {
    Experimental::nccl::Impl::recv(h.space(), sv, peer, h.comm());
    return Req<KokkosComm::Experimental::Nccl>(h.space().cuda_stream());
  }
};
#endif

}  // namespace KokkosComm::Impl
