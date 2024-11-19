// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace KokkosComm {

namespace Impl {
//
// fallback - most types are not a KokkosComm transport
template <typename T>
struct is_communication_space : public std::false_type {};

template <typename T>
struct is_reduction_operator : public std::false_type {};

}  // namespace Impl

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename T>
concept CommunicationSpace = KokkosComm::Impl::is_communication_space<T>::value;

template <typename T>
concept ReductionOperator = KokkosComm::Impl::is_reduction_operator<T>::value;

}  // namespace KokkosComm
