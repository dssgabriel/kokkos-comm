// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace KokkosComm {
namespace Impl {

/// Fallback: most types are not a KokkosComm communication space
template <typename T>
struct is_communication_space : public std::false_type {};
template <typename T>
inline constexpr bool is_communication_space_v = is_communication_space<T>::value;

/// Fallback: most types are not a KokkosComm reduction operator
template <typename T>
struct is_reduction_operator : public std::false_type {};
template <typename T>
inline constexpr bool is_reduction_operator_v = is_reduction_operator<T>::value;

}  // namespace Impl

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept MutKokkosView = KokkosView<T> && std::is_same_v<typename T::value_type, typename T::non_const_value_type>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename T>
concept KokkosMemorySpace = Kokkos::is_memory_space_v<T>;

template <typename T>
concept CommunicationSpace = requires {
  Impl::is_communication_space_v<T>;
  typename T::communication_space;
  typename T::execution_space;
  typename T::memory_space;
  typename T::communicator_type;
  typename T::request_type;
  typename T::datatype_type;
  typename T::reduction_op_type;
  typename T::size_type;
  typename T::rank_type;
};

template <typename T>
concept ReductionOperator = Impl::is_reduction_operator_v<T>;

template <typename Strategy, typename Comm, typename Exec, typename View>
concept PackingStrategy = CommunicationSpace<Comm> and KokkosExecutionSpace<Exec> and KokkosView<View> and requires(
    const Exec& exec,
    const View& active,
    std::optional<typename Strategy::packed_view_type>& packed,
    typename Comm::size_type& count,
    typename Comm::datatype_type& datatype
) {
  requires KokkosView<typename Strategy::packed_view_type>;

  { Strategy::setup_for(exec, active, packed, count, datatype) } -> std::same_as<void>;
  { Strategy::pack(exec, *packed, active) } -> std::same_as<void>;
  { Strategy::unpack(exec, active, *packed) } -> std::same_as<void>;
};

}  // namespace KokkosComm
