// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

namespace KokkosComm::Impl {

template <KokkosView View>
using default_contiguous_layout_t = std::conditional_t<
    std::is_same_v<typename View::array_layout, Kokkos::LayoutStride>,
    Kokkos::LayoutRight,
    typename View::array_layout>;

template <
    KokkosView View,
    KokkosMemorySpace Mem = typename View::memory_space,
    class Layout          = default_contiguous_layout_t<View>>
struct ContiguousView {
  using type = Kokkos::View<typename View::non_const_data_type, Layout, Mem>;
};

template <
    KokkosView View,
    KokkosMemorySpace Mem = typename View::memory_space,
    class Layout          = default_contiguous_layout_t<View>>
using contiguous_view_t = ContiguousView<View, Mem, Layout>::type;

/// @brief Allocate a contiguous View suitable for packing a non-contiguous View.
/// @tparam Exec A Kokkos Execution Space type.
/// @tparam View A Kokkos View type.
/// @param exec The execution space instance in which to perform the view allocation.
/// @param v The View to make a suitable contiguous allocation for.
/// @param label The label to give to the allocated contiguous View. Defaults to "contiguous_view".
template <
    KokkosExecutionSpace Exec,
    KokkosView View,
    KokkosMemorySpace Mem = typename View::memory_space,
    class Layout          = default_contiguous_layout_t<View>>
auto allocate_for(const Exec& exec, const std::string& label, const View& v) {
  using view_t = Kokkos::View<typename View::non_const_data_type, Layout, Mem>;

  // Unpack `v` extents into the `ContigView` constructor
  return [&label, &exec, &v ]<size_t... Is>(std::index_sequence<Is...>) {
    if constexpr (Kokkos::SpaceAccessibility<Exec, Mem>::accessible) {
      return view_t(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label), v.extent(Is)...);
    } else {
      return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), v.extent(Is)...);
    }
  }
  (std::make_index_sequence<rank<View>()>{});
}

template <
    KokkosExecutionSpace Exec,
    KokkosView View,
    KokkosMemorySpace Mem = typename View::memory_space,
    class Layout          = default_contiguous_layout_t<View>>
auto allocate_contiguous_for(const Exec& exec, const std::string& label, const View& v) {
  return allocate_for<Exec, View, Mem, Layout>(exec, label, v);
}

}  // namespace KokkosComm::Impl
