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

namespace KokkosComm::Experimental::nccl {

template <KokkosView SendV, KokkosView RecvV>
auto sendrecv(Communicator<NcclSpace, Kokkos::Cuda>& comm, const SendV& sv, int dst, const RecvV& rv, int src)
    -> Request<NcclSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::nccl::sendrecv");

  const auto exec = comm.exec();
  Request<NcclSpace> req;

  // TODO: Refactor once PackTraits returns packing strategy refactor lands.
  // Send/Recv sides are independent and this 4-way branch disappears.
  if (is_contiguous(sv) and is_contiguous(rv)) {
    KC_NCCL_CHECK(ncclGroupStart());
    KC_NCCL_CHECK(ncclSend(data_handle(sv), span(sv), datatype_for<NcclSpace>(sv), dst, comm.comm(), exec.cuda_stream())
    );
    KC_NCCL_CHECK(ncclRecv(data_handle(rv), span(rv), datatype_for<NcclSpace>(rv), src, comm.comm(), exec.cuda_stream())
    );
    KC_NCCL_CHECK(ncclGroupEnd());
  } else if (!is_contiguous(sv) and is_contiguous(rv)) {
    using SendPckr = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv   = SendPckr::pack(exec, "pckd_sv", sv);
    KC_NCCL_CHECK(ncclGroupStart());
    KC_NCCL_CHECK(
        ncclSend(data_handle(pckd_sv.view_), pckd_sv.count_, pckd_sv.datatype_, dst, comm.comm(), exec.cuda_stream())
    );
    KC_NCCL_CHECK(ncclRecv(data_handle(rv), span(rv), datatype_for<NcclSpace>(rv), src, comm.comm(), exec.cuda_stream())
    );
    KC_NCCL_CHECK(ncclGroupEnd());
    req.extend_view_lifetime(pckd_sv.view_);
  } else if (is_contiguous(sv) and !is_contiguous(rv)) {
    using RecvPckr = typename Impl::PackTraits<RecvV>::packer_type;
    auto pckd_rv   = RecvPckr::allocate_packed_for(exec, "pckd_rv", rv);
    KC_NCCL_CHECK(ncclGroupStart());
    KC_NCCL_CHECK(ncclSend(data_handle(sv), span(sv), datatype_for<NcclSpace>(sv), dst, comm.comm(), exec.cuda_stream())
    );
    KC_NCCL_CHECK(
        ncclRecv(data_handle(pckd_rv.view_), pckd_rv.count_, pckd_rv.datatype_, src, comm.comm(), exec.cuda_stream())
    );
    KC_NCCL_CHECK(ncclGroupEnd());
    // Implicitly extends `pckd_rv.view` and `rv` lifetime due to lambda capture
    req.add_callback([exec, rv, pckd_rv]() {
      Packer::unpack_into(exec, rv, pckd_rv.view_);
      exec.fence("fence unpacking after ncclRecv");
    });
  } else if (!is_contiguous(sv) and !is_contiguous(rv)) {
    using SendPckr = typename Impl::PackTraits<SendV>::packer_type;
    using RecvPckr = typename Impl::PackTraits<RecvV>::packer_type;
    auto pckd_sv   = SendPckr::pack(exec, "pckd_sv", sv);
    auto pckd_rv   = RecvPckr::allocate_packed_for(exec, "pckd_rv", rv);
    KC_NCCL_CHECK(ncclGroupStart());
    KC_NCCL_CHECK(
        ncclSend(data_handle(pckd_sv.view_), pckd_sv.count_, pckd_sv.datatype_, dst, comm.comm(), exec.cuda_stream())
    );
    KC_NCCL_CHECK(
        ncclRecv(data_handle(pckd_rv.view_), pckd_rv.count_, pckd_rv.datatype_, src, comm.comm(), exec.cuda_stream())
    );
    KC_NCCL_CHECK(ncclGroupEnd());
    req.extend_view_lifetime(pckd_sv.view_);
    // Implicitly extends `pckd_rv.view` and `rv` lifetime due to lambda capture
    req.add_callback([exec, rv, pckd_rv]() {
      Packer::unpack_into(exec, rv, pckd_rv.view_);
      exec.fence("fence unpacking after ncclRecv");
    });
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace KokkosComm::Experimental::nccl
