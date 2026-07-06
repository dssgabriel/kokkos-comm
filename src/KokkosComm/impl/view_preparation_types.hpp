// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <optional>
#include <type_traits>
#include <utility>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/traits.hpp>

namespace KokkosComm::Impl {

enum class ViewAccess { Read, Write, ReadWrite };

// State machine step tags
struct User {};
struct Staged {};
struct Packed {};
struct Ready {};

/// Direct user-provided View wrapper.
template <KokkosView View>
struct UserPayload {
  View view;
};

/// Staged View wrapper. Holds the original View, and a "maybe-staged" View.
/// Which one is "active" depends on whether the optional is null or not.
template <KokkosView View, KokkosView ActiveView>
struct StagedPayload {
  View view;
  std::optional<ActiveView> staged;

  [[nodiscard]] auto active() noexcept -> ActiveView& requires std::is_same_v<View, ActiveView> {
    return staged.has_value() ? *staged : view;
  }
  [[nodiscard]] auto active() const noexcept -> const ActiveView& requires std::is_same_v<View, ActiveView> {
    return staged.has_value() ? *staged : view;
  }

  [[nodiscard]] auto active() noexcept -> ActiveView& requires(!std::is_same_v<View, ActiveView>) { return *staged; }
  [[nodiscard]] auto active() const noexcept -> const ActiveView& requires(!std::is_same_v<View, ActiveView>) {
    return *staged;
  }
};

template <KokkosView View>
StagedPayload(View, std::nullopt_t) -> StagedPayload<View, View>;

template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView View, KokkosView ActiveView, typename Strat>
requires PackingStrategy<Strat, Comm, Exec, ActiveView>
struct ReadyPayload;

template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView View, KokkosView ActiveView, typename Strat>
requires PackingStrategy<Strat, Comm, Exec, ActiveView>
struct PackedPayload {
  // Only the Ready state may expose the communication buffer pointer.
  friend struct ReadyPayload<Comm, Exec, View, ActiveView, Strat>;

  StagedPayload<View, ActiveView> prev;
  std::optional<typename Strat::packed_view_type> packed;
  typename Comm::size_type count;
  typename Comm::datatype_type datatype;

  explicit PackedPayload(StagedPayload<View, ActiveView>&& staged)
      : prev(std::move(staged)),
        packed(std::nullopt),
        count(static_cast<typename Comm::size_type>(span(prev.active()))),
        datatype(datatype_for<Comm>(prev.active())) {}

  [[nodiscard]] auto is_packed() const noexcept -> bool { return packed.has_value(); }

 private:
  [[nodiscard]] auto buf_ptr() const noexcept -> void* {
    if (packed.has_value()) {
      return const_cast<void*>(static_cast<const void*>(data_handle(*packed)));
    }
    return const_cast<void*>(static_cast<const void*>(data_handle(prev.active())));
  }
};

template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView View, KokkosView ActiveView, typename Strat>
requires PackingStrategy<Strat, Comm, Exec, ActiveView>
struct ReadyPayload {
  PackedPayload<Comm, Exec, View, ActiveView, Strat> payload;

  [[nodiscard]] auto buf_ptr() const noexcept -> void* { return payload.buf_ptr(); }
  [[nodiscard]] auto count() const noexcept -> typename Comm::size_type { return payload.count; }
  [[nodiscard]] auto datatype() const noexcept -> typename Comm::datatype_type { return payload.datatype; }
};

template <typename State, typename Payload>
struct CommPayload {
  Payload payload;

  explicit CommPayload(Payload&& p) : payload(std::move(p)) {}

  CommPayload(const CommPayload&)                    = delete;
  auto operator=(const CommPayload&) -> CommPayload& = delete;
  CommPayload(CommPayload&&)                         = default;
  auto operator=(CommPayload&&) -> CommPayload&      = default;

  [[nodiscard]] auto buf_ptr() const noexcept requires std::is_same_v<State, Ready> { return payload.buf_ptr(); }

  [[nodiscard]] auto count() const noexcept requires std::is_same_v<State, Ready> { return payload.count(); }

  [[nodiscard]] auto datatype() const noexcept requires std::is_same_v<State, Ready> { return payload.datatype(); }
};

template <KokkosView View>
using UserState = CommPayload<User, UserPayload<View>>;

template <KokkosView View, KokkosView ActiveView>
using StagedState = CommPayload<Staged, StagedPayload<View, ActiveView>>;

template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView View, KokkosView ActiveView, typename Strat>
using PackedState = CommPayload<Packed, PackedPayload<Comm, Exec, View, ActiveView, Strat>>;

template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView View, KokkosView ActiveView, typename Strat>
using ReadyState = CommPayload<Ready, ReadyPayload<Comm, Exec, View, ActiveView, Strat>>;

}  // namespace KokkosComm::Impl
