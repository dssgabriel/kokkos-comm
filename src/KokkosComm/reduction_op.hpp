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

struct BAnd {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::BAnd> : public std::true_type {};

struct BOr {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::BOr> : public std::true_type {};

struct LAnd {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::LAnd> : public std::true_type {};

struct LOr {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::LOr> : public std::true_type {};

struct Max {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::Max> : public std::true_type {};

struct MaxLoc {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::MaxLoc> : public std::true_type {};

struct Min {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::Min> : public std::true_type {};

struct MinLoc {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::MinLoc> : public std::true_type {};

struct MinMax {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::MinMax> : public std::true_type {};

struct MinMaxLoc {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::MinMaxLoc> : public std::true_type {};

struct Sum {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::Sum> : public std::true_type {};

struct Prod {};
template <>
struct KokkosComm::Impl::is_reduction_operator<KokkosComm::Prod> : public std::true_type {};

struct Average {};
template <>
struct KokkosComm::Impl::is_reduction_operator<Average> : public std::true_type {};

#ifdef KOKKOSCOMM_ENABLE_NCCL
namespace Experimental::nccl::Impl {

template <ReductionOperator RedOp>
constexpr auto reduction_op() -> ncclRedOp_t {
  if constexpr (std::is_same_v<RedOp, Kokkos::Max>) {
    return ncclMax;
  } else if constexpr (std::is_same_v<RedOp, KokkosComm::Min>) {
    return ncclMin;
  } else if constexpr (std::is_same_v<RedOp, KokkosComm::Sum>) {
    return ncclSum;
  } else if constexpr (std::is_same_v<RedOp, KokkosComm::Prod>) {
    return ncclProd;
  } else if constexpr (std::is_same_v<RedOp, KokkosComm::Average>) {
    return ncclAvg;
  } else {
    static_assert(std::is_void_v<RedOp>, "nccl::reduction_op: operator not implemented");
    return ncclMax;  // unreachable
  }
}

template <ReductionOperator RedOp>
inline constexpr ncclRedOp_t reduction_op_v = reduction_op<RedOp>();

}  // namespace Experimental::nccl::Impl
#endif

}  // namespace KokkosComm
