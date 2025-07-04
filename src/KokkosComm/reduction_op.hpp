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
#ifdef KOKKOSCOMM_ENABLE_NCCL
#include <nccl.h>
#endif

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {

template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::BAnd> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::BOr> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::LAnd> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::LOr> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::Max> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::MaxLoc> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::Min> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::MinLoc> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::MinMax> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::MinMaxLoc> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::Sum> : public std::true_type {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Kokkos::Prod> : public std::true_type {};

/// Custom "marker" reducer for computing an average.
///
/// Does nothing on its own, but is meant to serve as a "marker" reduction operator when calling NCCL reductions.
struct Average {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Average> : public std::true_type {};

#ifdef KOKKOSCOMM_ENABLE_NCCL
namespace Experimental::nccl::Impl {

template <ReductionOperator RedOp>
constexpr auto reduction_op() -> ncclRedOp_t {
  if constexpr (std::is_same_v<RedOp, Kokkos::Max>) {
    return ncclMax;
  } else if constexpr (std::is_same_v<RedOp, Kokkos::Min>) {
    return ncclMin;
  } else if constexpr (std::is_same_v<RedOp, Kokkos::Sum>) {
    return ncclSum;
  } else if constexpr (std::is_same_v<RedOp, Kokkos::Prod>) {
    return ncclProd;
  } else if constexpr (std::is_same_v<RedOp, KokkosComm::Average>) {
    return ncclAvg;
  } else {
    static_assert(std::is_void_v<RedOp>, "NCCL reduction operator not implemented");
    return ncclMax;  // unreachable
  }
}

template <ReductionOperator RedOp>
inline constexpr ncclRedOp_t reduction_op_v = reduction_op<RedOp>();

}  // namespace Experimental::nccl::Impl
#endif

}  // namespace KokkosComm
