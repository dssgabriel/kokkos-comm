// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <optional>

#include <cuda.h>
#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>
#include "nccl_space.hpp"

namespace KokkosComm {

template <>
class Communicator<Kokkos::Cuda, Experimental::NcclSpace> {
 public:
  using execution_space     = Kokkos::Cuda;
  using communication_space = Experimental::NcclSpace;
  using communicator_type   = communication_space::handle_type;
  using size_type           = communication_space::size_type;

  [[nodiscard]] static auto split(
      const communicator_type comm,
      const execution_space& exec = Kokkos::DefaultExecutionSpace{},
      Color color,
      Key key,
      Rank root = 0_rank
  ) -> std::optional<Communicator<execution_space, MpiSpace>> {
    communicator_type new_comm;
    ncclCommSplit(comm, color, key, &new_comm, nullptr);
    if (color == NCCL_SPLIT_NOCOLOR) {
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
    return Communicator<MpiSpace>(exec, comm, root, is_owning);
  }

  ~Communicator() {
    // Free the underlying `ncclComm_t` if it is marked as owning
    if (is_owning) {
      ncclCommDestroy(comm_);
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

  auto set_size() noexcept -> void { ncclCommCount(comm_, &size_); }
  auto set_rank() noexcept -> void { ncclCommUserRank(comm_, &rank_.value); }

  execution_space exec_;
  communicator_type comm_;
  size_type size_;
  Rank rank_;
  Rank root_;
  bool is_owning_;
};

}  // namespace KokkosComm
