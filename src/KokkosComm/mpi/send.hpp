// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "comm_mode.hpp"
#include "communicator.hpp"
#include "isend.hpp"

#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm, SendMode) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::send");

  auto h = Communicator<MpiSpace, ExecSpace>::from_raw(comm, space);
  isend(h, sv, dest, tag, SendMode{}).wait();

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  send(space, sv, dest, tag, comm, DefaultCommMode{});
}

/// NOTE: This overload has the side effect of fencing on the default execution space.
template <KokkosView SendView>
void send(const SendView &sv, int dest, int tag, MPI_Comm comm) {
  send(Kokkos::DefaultExecutionSpace(), sv, dest, tag, comm, DefaultCommMode{});
}

}  // namespace KokkosComm::mpi
