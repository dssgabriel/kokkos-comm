// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/reduction_op.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
auto reduce(const ExecSpace& exec, const SendView& sv, RecvView& rv, ncclRedOp_t op, int root, ncclComm_t comm)
    -> Request<NcclSpace> {
  using ST         = typename SendView::non_const_value_type;
  using RT         = typename RecvView::non_const_value_type;
  using SendPacker = typename Impl::PackTraits<SendView>::packer_type;
  using RecvPacker = typename Impl::PackTraits<RecvView>::packer_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::reduce: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::reduce");

  const int rank = [=]() {
    int _r;
    ncclCommUserRank(comm, &_r);
    return _r;
  }();

  Request<NcclSpace> req;
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  constexpr auto dtype = datatype<NcclSpace, ST>();
  if (is_contiguous(sv)) {
    const auto count = span(sv);
    if (rank == root and not is_contiguous(rv)) {
      auto pckd_rv = RecvPacker::allocate_packed_for(exec, "pckd_rv", rv);
      ncclReduce(data_handle(sv), data_handle(pckd_rv.view_), count, dtype, op, root, comm, exec.cuda_stream());
      req.add_callback([exec, rv, pckd_rv]() {
        RecvPacker::unpack_into(exec, rv, pckd_rv.view_);
        exec.fence("fence `pckd_rv` unpacking after NCCL reduce");
      });
    } else {
      ncclReduce(data_handle(sv), data_handle(rv), count, dtype, op, root, comm, exec.cuda_stream());
    }
  } else {
    auto pckd_sv = SendPacker::pack(exec, "pckd_sv", sv);
    req.extend_view_lifetime(pckd_sv.view_);
    const auto count = pckd_sv.count_;
    if (rank == root and not is_contiguous(rv)) {
      auto pckd_rv = RecvPacker::allocate_packed_for(exec, "pckd_rv", rv);
      ncclReduce(
          data_handle(pckd_sv.view_), data_handle(pckd_rv.view_), count, dtype, op, root, comm, exec.cuda_stream()
      );
      req.add_callback([exec, rv, pckd_rv]() {
        RecvPacker::unpack_into(exec, rv, pckd_rv.view_);
        exec.fence("fence `pckd_rv` unpacking after NCCL reduce");
      });
    } else {
      ncclReduce(data_handle(pckd_sv.view_), data_handle(rv), count, dtype, op, root, comm, exec.cuda_stream());
    }
  }
  req.capture_stream_state(exec.cuda_stream());

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView, ReductionOperator RedOp>
struct Reduce<SendView, RecvView, RedOp, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& comm, const SendView sv, RecvView rv, int root)
      -> Request<NcclSpace> {
    return nccl::reduce(comm.exec(), sv, rv, reduction_op<NcclSpace, RedOp>(), root, comm.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
