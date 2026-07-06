// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <optional>

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/traits.hpp>

#include <KokkosComm/impl/contiguous.hpp>

namespace KokkosComm::Impl {

template <CommunicationSpace Comm, KokkosView View>
struct DeepCopy {
  using packed_view_type = contiguous_view_t<View>;

  template <KokkosExecutionSpace Exec>
  static auto setup_for(
      const Exec& exec,
      const View& src,
      std::optional<packed_view_type>& packed,
      typename Comm::size_type& count,
      typename Comm::datatype_type& datatype
  ) -> void {
    packed   = allocate_for(exec, "KokkosComm::Impl::DeepCopy::packed", src);
    count    = static_cast<typename Comm::size_type>(span(*packed));
    datatype = datatype_for<Comm>(*packed);
  }

  template <KokkosExecutionSpace Exec>
  static auto pack(const Exec& exec, const packed_view_type& dst, const View& src) -> void {
    Kokkos::deep_copy(exec, dst, src);
  }

  template <KokkosExecutionSpace Exec>
  static auto unpack(const Exec& exec, const View& dst, const packed_view_type& src) -> void {
    Kokkos::deep_copy(exec, dst, src);
  }
};

}  // namespace KokkosComm::Impl
