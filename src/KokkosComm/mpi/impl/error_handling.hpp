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

#include <iostream>
#include <mpi.h>

namespace KokkosComm::mpi {
inline void fail_if(bool condition, const char* error_msg) {
  if (condition) {
#ifdef KOKKOSCOMM_ABORT_ON_ERROR
    std::cerr << error_msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
    Kokkos::abort(error_msg);
#endif
  }
}
}  // namespace KokkosComm::mpi
