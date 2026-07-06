// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/impl/view_preparation.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

namespace KC = KokkosComm;

template <MutKokkosView View>
auto broadcast(const Kokkos::Cuda& space, View& v, int root, ncclComm_t comm) -> Request<NcclSpace> {
  static_assert(
      KC::rank<View>() <= 1,
      "KokkosComm::Experimental::nccl::broadcast: Views with rank higher than 1 are not supported"
  );
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::broadcast");

  Request<NcclSpace> req;
  auto ready = KC::Impl::prepare<KC::Impl::ViewAccess::ReadWrite>(space, v, req);
  ncclBcast(ready.buf_ptr(), ready.count(), ready.datatype(), root, comm, space.cuda_stream());
  req.capture_stream_state(space.cuda_stream());

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <MutKokkosView View>
struct Broadcast<View, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, View v, int root) -> Request<NcclSpace> {
    return nccl::broadcast(h.exec(), v, root, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
