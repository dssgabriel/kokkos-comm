// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <optional>
#include <type_traits>
#include <utility>

#include <KokkosComm/config.hpp>
#include <KokkosComm/concepts.hpp>
#include <KokkosComm/fwd.hpp>
#include <KokkosComm/traits.hpp>

#include "contiguous.hpp"
#include "view_preparation_types.hpp"

#ifdef KOKKOSCOMM_ENABLE_MPI_GPU_AWARE
#include <KokkosComm/mpi/mpi_space.hpp>
#endif

namespace KokkosComm::Impl {

// Staging is required when the view's memory space is not accessible from the backend's associated execution space.
template <CommunicationSpace Comm, KokkosView View>
struct needs_staging
    : std::bool_constant<
          !Kokkos::SpaceAccessibility<typename Comm::execution_space, typename View::memory_space>::accessible> {};

// Specialize for GPU-aware MPI, which can access regardless.
#ifdef KOKKOSCOMM_ENABLE_MPI_GPU_AWARE
template <KokkosView View>
struct needs_staging<MpiSpace, View> : std::false_type {};
#endif

template <CommunicationSpace Comm, KokkosView View>
inline constexpr bool needs_staging_v = needs_staging<Comm, View>::value;

template <typename Comm>
struct allows_staging : std::true_type {};

#if defined(KOKKOSCOMM_ENABLE_NCCL) && !defined(KOKKOSCOMM_ENABLE_NCCL_DEVICE_STAGING)
template <>
struct allows_staging<Experimental::NcclSpace> : std::false_type {};
#endif

template <CommunicationSpace Comm>
inline constexpr bool allows_staging_v = allows_staging<Comm>::value;

template <
    KokkosView View,
    KokkosMemorySpace Mem = typename View::memory_space,
    class Layout          = typename View::execution_space::array_layout>
struct StagingView {
  using type = Kokkos::View<typename View::non_const_data_type, Layout, Mem>;
};

template <
    KokkosView View,
    KokkosMemorySpace Mem = typename View::memory_space,
    class Layout          = typename View::execution_space::array_layout>
using staging_view_t = StagingView<View, Mem, Layout>::type;

template <KokkosExecutionSpace Exec, KokkosView DstView, KokkosView SrcView>
auto stage(const Exec& exec, const DstView& dst, const SrcView& src) -> void {
  Kokkos::deep_copy(exec, dst, src);
}

template <KokkosExecutionSpace Exec, KokkosView DstView, KokkosView SrcView>
auto unstage(const Exec& exec, const DstView& dst, const SrcView& src) -> void {
  Kokkos::deep_copy(exec, dst, src);
}

template <ViewAccess Access, CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView View>
[[nodiscard]] auto maybe_stage(const Exec& exec, UserState<View> state, Request<Comm>& req) {
  auto view = std::move(state.payload.view);
  // Default: extend the lifetime of the user view til communication completion.
  req.extend_view_lifetime(view);

  if constexpr (needs_staging_v<Comm, View>) {
    static_assert(
        allows_staging_v<Comm>,
        "KokkosComm: this View requires staging for the selected communication backend, but staging is disabled for "
        "that backend"
    );

    using StagingView = staging_view_t<View>;

    auto staged = allocate_for<Exec, View, typename Comm::memory_space, typename Comm::execution_space::array_layout>(
        exec, "KokkosComm::Impl::staged", view
    );
    req.extend_view_lifetime(staged);

    // Actually copy the memory in the allocated buffer if the View is read.
    if constexpr (Access == ViewAccess::Read || Access == ViewAccess::ReadWrite) {
      stage(exec, staged, view);
    }

    return StagedState<View, StagingView>{StagedPayload{view, std::optional<StagingView>{staged}}};
  } else {
    return StagedState<View, View>{StagedPayload{view, std::nullopt}};
  }
}

}  // namespace KokkosComm::Impl
