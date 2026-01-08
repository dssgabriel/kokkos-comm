// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>

namespace KokkosComm::Impl {

template <KokkosView V>
inline constexpr bool needs_staging_v =
    !Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename V::memory_space>::accessible;

/// Stage view on the host for non-GPU-aware communications.
/// No-op if `view` is device-accessible from the host.
template <KokkosExecutionSpace E, KokkosView V>
auto stage_for(const E& space, const V& view) -> typename V::host_mirror_type {
  if constexpr (needs_staging_v<V>) {
    return Kokkos::create_mirror_view_and_copy(space, view);
  } else {
    // For views already in the host memory space, this only creates a reference.
    // For views in a device memory space that is accessible from the host (e.g. UVM), this should not allocate any
    // memory, only create a reference cast to the `HostMirror` view type.
    return typename V::host_mirror_type(view);
  }
}

/// Copy back to device (e.g. for receive operations).
/// No-op if `dst` is device-accessible from the host.
template <KokkosExecutionSpace E, KokkosView V>
auto copy_back(const E& space, const V& dst, const typename V::host_mirror_type& src) -> void {
  if constexpr (needs_staging_v<V>) {
    Kokkos::deep_copy(space, dst, src);
  }
}

}  // namespace KokkosComm::Impl
