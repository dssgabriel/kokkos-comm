// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::Experimental::nccl {

template <KokkosExecutionSpace Exec, KokkosView SendV>
auto send(Communicator<NcclSpace, Exec>& comm, const SendV& sv, int dst) -> Request<NcclSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::nccl::send");

  const auto exec = comm.exec();
  Request<NcclSpace> req;
  if (is_contiguous(sv)) {
    KC_NCCL_CHECK(ncclSend(data_handle(sv), span(sv), datatype_for<NcclSpace>(sv), dst, comm.comm(), exec.cuda_stream())
    );
  } else {
    using Packer = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv = Packer::pack(exec, "pckd_sv", sv);
    // No need to fence since we enqueue the packing on the same execution space
    KC_NCCL_CHECK(
        ncclSend(data_handle(pckd_sv.view_), pckd_sv.count_, pckd_sv.datatype_, dst, comm.comm(), exec.cuda_stream())
    );
    req.capture_stream_state(exec.cuda_stream());
    req.extend_view_lifetime(pckd_sv.view_);
  }
  req.extend_view_lifetime(sv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace KokkosComm::Experimental::nccl
