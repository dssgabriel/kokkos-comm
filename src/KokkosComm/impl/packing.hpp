// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>
#include <utility>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/config.hpp>
#include <KokkosComm/fwd.hpp>
#include <KokkosComm/traits.hpp>

#include "view_preparation_types.hpp"
#include "packing_strategy/deep_copy.hpp"

#if defined(KOKKOSCOMM_ENABLE_MPI)
#include <mpi.h>
#endif

#if defined(KOKKOSCOMM_ENABLE_MPI) && defined(KOKKOSCOMM_ENABLE_MPI_DERIVED_DATATYPE_PACKING)
#include <KokkosComm/impl/packing_strategy/mpi_derived_datatype.hpp>
#include <KokkosComm/mpi/mpi_space.hpp>
#endif

namespace KokkosComm::Impl {

template <CommunicationSpace Comm, KokkosView View>
struct default_packing_strategy {
  using type = DeepCopy<Comm, View>;
};

#if defined(KOKKOSCOMM_ENABLE_MPI) && defined(KOKKOSCOMM_ENABLE_MPI_DERIVED_DATATYPE_PACKING)
template <KokkosView View>
struct default_packing_strategy<MpiSpace, View> {
  using type = KokkosComm::mpi::Impl::DerivedDatatype<MpiSpace, View>;
};
#endif

template <CommunicationSpace Comm, KokkosView View>
using default_packing_strategy_t = typename default_packing_strategy<Comm, View>::type;

template <
    ViewAccess Access,
    typename Strat,
    CommunicationSpace Comm,
    KokkosExecutionSpace Exec,
    KokkosView View,
    KokkosView ActiveView>
requires PackingStrategy<Strat, Comm, Exec, ActiveView>
[[nodiscard]] auto maybe_pack(const Exec& exec, StagedState<View, ActiveView> state, Request<Comm>& req)
    -> PackedState<Comm, Exec, View, ActiveView, Strat> {
  auto payload = PackedPayload<Comm, Exec, View, ActiveView, Strat>{std::move(state.payload)};
  auto& active = payload.prev.active();

  if (!is_contiguous(active)) {
    Strat::setup_for(exec, active, payload.packed, payload.count, payload.datatype);

#if defined(KOKKOSCOMM_ENABLE_MPI)
    if constexpr (requires { typename Strat::mpi_derived_datatype_strategy_tag; }) {
      auto dtype = payload.datatype;
      req.add_callback([dtype]() mutable { MPI_Type_free(&dtype); });
    }
#endif

    if constexpr (Access == ViewAccess::Read || Access == ViewAccess::ReadWrite) {
      Strat::pack(exec, *payload.packed, active);
    }
  }

  if (payload.packed.has_value()) {
    req.extend_view_lifetime(*payload.packed);
  } else {
    req.extend_view_lifetime(active);
  }

  if constexpr (Access == ViewAccess::Write || Access == ViewAccess::ReadWrite) {
    auto original            = payload.prev.view;
    auto staged              = payload.prev.staged;
    auto active_dst          = payload.prev.active();
    auto packed_for_callback = payload.packed;

    req.add_callback([exec, original, staged, active_dst, packed_for_callback]() mutable {
      if (packed_for_callback.has_value()) {
        Strat::unpack(exec, active_dst, *packed_for_callback);
      }
      if (staged.has_value()) {
        unstage(exec, original, *staged);
      }
      // Unpack/unstage may enqueue asynchronous deep copies; wait before marking the request callbacks complete.
      exec.fence("KokkosComm::Impl::maybe_pack write-back");
    });
  }

  return PackedState<Comm, Exec, View, ActiveView, Strat>{std::move(payload)};
}

}  // namespace KokkosComm::Impl
