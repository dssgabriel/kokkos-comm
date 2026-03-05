// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <optional>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>
#include "mpi_space.hpp"

namespace KokkosComm {

template <KokkosExecutionSpace Ex>
class Communicator<Ex, MpiSpace> {
 public:
  using execution_space     = Ex;
  using communication_space = MpiSpace;
  using communicator_type   = communication_space::communicator_type;
  using size_type           = int;

  [[nodiscard]] static auto world(const execution_space& exec = Kokkos::DefaultExecutionSpace{}, Rank root = 0_rank)
      -> Communicator<execution_space, MpiSpace> {
    return Communicator<MpiSpace>(exec, MPI_COMM_WORLD, root, false);
  }

  [[nodiscard]] static auto duplicate(
      const communicator_type comm, const execution_space& exec = Kokkos::DefaultExecutionSpace{}, Rank root = 0_rank
  ) -> Communicator<execution_space, MpiSpace> {
    communicator_type new_comm;
    MPI_Comm_dup(comm, &new_comm);
    return Communicator<MpiSpace>(exec, new_comm, root, true);
  }

  [[nodiscard]] static auto split(
      const communicator_type comm,
      const execution_space& exec = Kokkos::DefaultExecutionSpace{},
      Color color,
      Key key,
      Rank root = 0_rank
  ) -> std::optional<Communicator<execution_space, MpiSpace>> {
    communicator_type new_comm;
    MPI_Comm_split(comm, color, key, &new_comm);
    if (color == MPI_UNDEFINED) {
      return std::nullopt;
    }
    return Communicator<MpiSpace>(exec, new_comm, root, true);
  }

  [[nodiscard]] static auto try_from_raw(
      communicator_type comm,
      const execution_space& exec = Kokkos::DefaultExecutionSpace{},
      Rank root                   = 0_rank,
      bool is_owning              = false
  ) -> std::optional<Communicator<execution_space, MpiSpace>> {
    if (comm == MPI_COMM_NULL) {
      return std::nullopt;
    }
    return Communicator<MpiSpace>(exec, comm, root, is_owning);
  }

  ~Communicator() {
    // Free the underlying `MPI_Comm` if it is marked as owning, and is not `MPI_COMM_WORLD`, nor `MPI_COMM_SELF`
    if (is_owning && !is_world() && !is_self()) {
      MPI_Comm_free(comm_);
    }
  }

  [[nodiscard]] constexpr auto exec() const noexcept -> const execution_space& { return exec_; }
  [[nodiscard]] constexpr auto exec() noexcept -> execution_space& { return exec_; }
  [[nodiscard]] constexpr auto comm() const noexcept -> const communicator_type& { return comm_; }
  [[nodiscard]] constexpr auto comm() noexcept -> communicator_type& { return comm_; }
  [[nodiscard]] constexpr auto size() const noexcept -> size_type { return size_; }
  [[nodiscard]] constexpr auto rank() const noexcept -> Rank { return rank_; }
  [[nodiscard]] constexpr auto root() const noexcept -> Rank { return rank_; }

  constexpr auto set_root(Rank root) noexcept -> void { root_ = root; }

 private:
  explicit Communicator(const execution_space& exec, communicator_type comm, Rank root, bool is_owning)
      : exec_(exec), comm_(comm), root_(root), is_owning_(is_owning) {
    set_size();
    set_rank();
  }

  auto is_world() const noexcept -> bool { return comm_ == MPI_COMM_WORLD; }
  auto is_self() const noexcept -> bool { return comm_ == MPI_COMM_SELF; }

  auto set_size() noexcept -> void { MPI_Comm_size(comm_, &size_); }
  auto set_rank() noexcept -> void { MPI_Comm_rank(comm_, &rank_.value); }

  execution_space exec_;
  communicator_type comm_;
  size_type size_;
  Rank rank_;
  Rank root_;
  bool is_owning_;
};

}  // namespace KokkosComm
