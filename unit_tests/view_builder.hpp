// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>

namespace Impl {

// --- Type-level pointer builder: T* -> T** -> T*** ... ---
template <typename T, std::size_t N>
struct AddPtrs {
  using type = typename AddPtrs<T*, N - 1>::type;
};
template <typename T>
struct AddPtrs<T, 0> {
  using type = T;
};
// Rank stars: AddPtrs<T, N>::type == T followed by N stars
template <typename T, std::size_t N>
using Stars = typename AddPtrs<T, N>::type;

// --- Index sequence helpers ---
// Build a tuple of (Kokkos::ALL, ..., Kokkos::ALL, 1) for subview
template <std::size_t... Is>
auto make_all_tuple(std::index_sequence<Is...>) {
  // Is... expands to N copies of Kokkos::ALL, then append 1
  return std::make_tuple((void(Is), Kokkos::ALL)..., 1);
}

template <std::size_t Rank>
auto all_tuple() {
  return make_all_tuple(std::make_index_sequence<Rank>{});
}

// Apply subview with a tuple of args
template <typename ViewType, typename Tuple, std::size_t... Is>
auto apply_subview(ViewType& v, Tuple&& t, std::index_sequence<Is...>) {
  return Kokkos::subview(v, std::get<Is>(std::forward<Tuple>(t))...);
}

}  // namespace Impl

struct Contig {};
struct NonContig {};

template <typename T, std::size_t Rank>
struct ViewBuilder {
  // Contiguous: View<T** (Rank stars)>(name, e0, ..., eN-1)
  template <typename... Extents>
  static auto view(Contig, const std::string& name, Extents... exts) {
    static_assert(sizeof...(exts) == Rank);
    return Kokkos::View<Impl::Stars<T, Rank>>(name, exts...);
  }

  // Non-contiguous: LayoutRight view with extra trailing extent=2,
  // then subview(ALL, ..., ALL, 1) to pick column 1.
  template <typename... Extents>
  static auto view(NonContig, const std::string& name, Extents... exts) {
    static_assert(sizeof...(exts) == Rank);
    // View type has Rank+1 stars, LayoutRight
    Kokkos::View<Impl::Stars<T, Rank + 1>, Kokkos::LayoutRight> v(name, exts..., 2);
    // Subview args: ALL x Rank, then 1
    auto args = Impl::all_tuple<Rank>();
    return Impl::apply_subview(v, args, std::make_index_sequence<Rank + 1>{});
  }
};
