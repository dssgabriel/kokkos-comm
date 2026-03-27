// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

struct Contig {};
struct NonContig {};

namespace Impl {

template <typename T, size_t N>
struct AddPtrs {
  using type = typename AddPtrs<T*, N - 1>::type;
};
template <typename T>
struct AddPtrs<T, 0> {
  using type = T;
};
template <typename T, size_t N>
using Stars = typename AddPtrs<T, N>::type;

template <size_t Rank>
auto all_then_one() {
  return [&]<size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple((void(Is), Kokkos::ALL)..., 1);
  }(std::make_index_sequence<Rank>{});
}
template <typename ViewType, typename Tuple, size_t... Is>
auto apply_subview(ViewType& v, Tuple&& t, std::index_sequence<Is...>) {
  return Kokkos::subview(v, std::get<Is>(std::forward<Tuple>(t))...);
}

}  // namespace Impl

template <typename T, size_t R, typename... Extents>
auto build_view(Contig, const std::string& name, Extents... exts) {
  static_assert(sizeof...(exts) == R, "Number of extents must match Rank");
  return Kokkos::View<Impl::Stars<T, R>>(name, exts...);
}
template <typename T, size_t R, typename... Extents>
auto build_view(NonContig, const std::string& name, Extents... exts) {
  static_assert(sizeof...(exts) == R, "Number of extents must match Rank");
  Kokkos::View<Impl::Stars<T, R + 1>, Kokkos::LayoutRight> v(name, exts..., 2);
  auto args = Impl::all_then_one<R>();
  return Impl::apply_subview(v, args, std::make_index_sequence<R + 1>{});
}

