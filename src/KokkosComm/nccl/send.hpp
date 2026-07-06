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

#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Experimental::nccl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
auto send(const ExecSpace& space, const SendView& sv, int peer, ncclComm_t comm) -> Request<NcclSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  Request<NcclSpace> req;
  auto ready = KokkosComm::Impl::prepare<KokkosComm::Impl::ViewAccess::Read>(space, sv, req);
  KC_NCCL_CHECK(ncclSend(ready.buf_ptr(), ready.count(), ready.datatype(), peer, comm, space.cuda_stream()));
  req.capture_stream_state(space.cuda_stream());

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::nccl
namespace Impl {

template <KokkosView SendView>
struct Send<SendView, Kokkos::Cuda, Experimental::NcclSpace> {
  static auto execute(Communicator<Experimental::NcclSpace, Kokkos::Cuda>& h, SendView sv, int peer)
      -> Request<Experimental::NcclSpace> {
    return Experimental::nccl::send(h.exec(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
