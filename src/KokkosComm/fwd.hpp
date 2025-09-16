<<<<<<< HEAD
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/config.hpp>
#include "concepts.hpp"
#include "reduction_op.hpp"
=======
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#pragma once

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/config.hpp>
#include <KokkosComm/reduction_op.hpp>
>>>>>>> 774232e (refactor(nccl)!: enabling NCCL also forward-declares MPI)

namespace KokkosComm {

// NCCL backend also implicitly declares MPI
#if defined(KOKKOSCOMM_ENABLE_NCCL)
class Mpi;
class Nccl;
using DefaultCommunicationSpace  = Nccl;
using FallbackCommunicationSpace = Mpi;
#elif defined(KOKKOSCOMM_ENABLE_MPI)
class Mpi;
using DefaultCommunicationSpace  = Mpi;
using FallbackCommunicationSpace = Mpi;
#else
#error at least one communication space must be enabled
#endif

template <CommunicationSpace CommSpace = DefaultCommunicationSpace>
class Req;

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
class Handle;

namespace Impl {

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct Recv;

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct Send;

}  // namespace Impl

// Collectives are currently experimental functions
namespace Experimental::Impl {

template <KokkosView SendView, KokkosView RecvView = SendView,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Broadcast;

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct AllToAll;

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct AllGather;

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct AllReduce;

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Reduce;

}  // namespace Experimental::Impl

}  // namespace KokkosComm
