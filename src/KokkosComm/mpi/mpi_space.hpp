// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {

/// The MPI communication space.
struct MpiSpace {
  using communication_space = MpiSpace;
  using communicator_type   = MPI_Comm;
  using request_type        = MPI_Request;
  using datatype_type       = MPI_Datatype;
  using reduction_op_type   = MPI_Op;
  using size_type           = int;
  using rank_type           = int;
  using execution_space     = Kokkos::DefaultHostExecutionSpace;
  using memory_space        = typename execution_space::memory_space;
};

// KokkosComm::MpiSpace is a KokkosComm::CommunicationSpace
template <>
struct Impl::is_communication_space<MpiSpace> : public std::true_type {};

}  // namespace KokkosComm
