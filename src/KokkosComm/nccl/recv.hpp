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

/// @brief Initiates a non-blocking receive operation.
/// @tparam Exec A Kokkos execution space type.
/// @tparam RecvV A Kokkos View type.
/// @param comm The communicator handle associated with the operation.
/// @param rv The View to receive into.
/// @param src The source rank.
/// @returns A Request object representing the non-blocking receive operation.
template <KokkosExecutionSpace Exec, KokkosView RecvV>
auto recv(Communicator<NcclSpace, Exec>& comm, const RecvV& rv, int src) -> Request<NcclSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::recv");

  const auto exec = comm.exec();
  Request<NcclSpace> req;
  if (is_contiguous(rv)) {
    KC_NCCL_CHECK(ncclRecv(data_handle(rv), span(rv), datatype_for<NcclSpace>(rv), src, comm.comm(), exec.cuda_stream())
    );
  } else {
    using Packer = typename Impl::PackTraits<RecvV>::packer_type;
    auto pckd_rv = Packer::allocate_packed_for(exec, "pckd_rv", rv);
    // No need to fence since we enqueue the packed allocation on the same execution space
    KC_NCCL_CHECK(
        ncclRecv(data_handle(pckd_rv.view_), pckd_rv.count_, pckd_rv.datatype_, src, comm.comm(), exec.cuda_stream())
    );
    req.add_callback([exec, rv, pckd_rv]() {
      Packer::unpack_into(exec, rv, pckd_rv.view_);
      exec.fence("fence unpacking after ncclRecv");
    });
  }
  req.capture_stream_state(exec.cuda_stream());
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace KokkosComm::Experimental::nccl
