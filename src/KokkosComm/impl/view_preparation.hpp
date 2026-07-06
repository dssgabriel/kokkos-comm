// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>
#include <utility>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/fwd.hpp>

#include "packing.hpp"
#include "staging.hpp"
#include "view_preparation_types.hpp"

/// View preparation as a state machine.
///
/// ╔════════════╗
/// ║    USER    ║
/// ╚════════════╝
///       │
///  maybe_stage
///       │
///       ▼
/// ╔════════════╗
/// ║   STAGED   ║
/// ╚════════════╝
///       │
///  maybe_pack
///       │
///       ▼
/// ╔════════════╗
/// ║   PACKED   ║
/// ╚════════════╝
///       │
///   make_ready
///       │
///       ▼
/// ╔════════════╗
/// ║   READY    ║
/// ╚════════════╝

namespace KokkosComm::Impl {
namespace {

template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView View, KokkosView ActiveView, typename Strat>
[[nodiscard]] auto make_ready(PackedState<Comm, Exec, View, ActiveView, Strat> state) {
  return ReadyState<Comm, Exec, View, ActiveView, Strat>{
      ReadyPayload<Comm, Exec, View, ActiveView, Strat>{std::move(state.payload)}};
}

}  // namespace

/// @brief Prepare a user View for communication.
///
/// Transitions a user-provided View through staging and packing, then returns a Ready state exposing the backend
/// communication buffer pointer, count, and datatype.
///
/// @tparam Access Requested access mode for the communication operation.
/// @tparam Exec Execution space used for staging and packing operations.
/// @tparam View User-provided Kokkos View type.
/// @tparam Comm Communication backend space, inferred from the request.
///
/// @param exec Execution space instance used to enqueue staging and packing work.
/// @param view User-provided View to prepare. Write access modes require mutable Views.
/// makes the view handle const.
/// @param req Backend request whose lifetime extensions and completion callbacks are updated by preparation.
///
/// @return Ready state containing the communication-ready payload.
template <ViewAccess Access, KokkosExecutionSpace Exec, KokkosView View, CommunicationSpace Comm>
requires(Access == ViewAccess::Read || MutKokkosView<View>)
    [[nodiscard]] auto prepare(const Exec& exec, const View& view, Request<Comm>& req) {
  // Simple type conversion to the initial state of the view preparation state machine
  auto user = UserState<View>{UserPayload{view}};

  // Potentially stage the view.
  auto staged = maybe_stage<Access>(exec, std::move(user), req);

  using ActiveView   = std::remove_cvref_t<decltype(staged.payload.active())>;
  using PackingStrat = default_packing_strategy_t<Comm, ActiveView>;
  // Potentially pack the staged view.
  auto packed = maybe_pack<Access, PackingStrat>(exec, std::move(staged), req);

  // Make it ready, simple type conversion to the final state of the view preparation state machine
  return make_ready(std::move(packed));
}

}  // namespace KokkosComm::Impl
