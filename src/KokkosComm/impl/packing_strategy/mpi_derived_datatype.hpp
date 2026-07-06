// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>

#include <mpi.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/mpi/mpi_space.hpp>
#include <KokkosComm/traits.hpp>

namespace KokkosComm::mpi::Impl {

template <CommunicationSpace Comm, KokkosView View>
requires std::is_same_v<Comm, MpiSpace>
struct DerivedDatatype {
  using mpi_derived_datatype_strategy_tag = void;
  using packed_view_type                  = View;

  template <KokkosExecutionSpace Exec>
  static auto setup_for(
      const Exec& /*exec*/,
      const View& src,
      std::optional<packed_view_type>& packed,
      typename Comm::size_type& count,
      typename Comm::datatype_type& dtype
  ) -> void {
    using value_type = typename View::value_type;

    MPI_Datatype type = datatype<Comm, value_type>();
    for (std::size_t d = 0; d < KokkosComm::rank(src); ++d) {
      MPI_Datatype newtype;
      MPI_Type_create_hvector(
          KokkosComm::extent(src, d), 1, KokkosComm::stride(src, d) * sizeof(value_type), type, &newtype
      );
      type = newtype;
    }
    MPI_Type_commit(&type);

    packed = src;
    count  = 1;
    dtype  = type;
  }

  template <KokkosExecutionSpace Exec>
  static auto pack(const Exec& /*exec*/, const packed_view_type& /*dst*/, const View& /*src*/) -> void {}

  template <KokkosExecutionSpace Exec>
  static auto unpack(const Exec& /*exec*/, const View& /*dst*/, const packed_view_type& /*src*/) -> void {}
};

}  // namespace KokkosComm::mpi::Impl
