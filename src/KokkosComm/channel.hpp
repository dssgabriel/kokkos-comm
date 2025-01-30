//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
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

#include <Kokkos_Core.hpp>

#include "KokkosComm/fwd.hpp"
//#include "commmode.hpp"
//#include "impl/pack_traits.hpp"
//#include "impl/include_mpi.hpp"

namespace KokkosComm {

    //  TODO: add Channel<CommSpace> object
    //      see: Req<CommSpace> ✧･ﾟ:✧･ﾟ 
    template <class CommSpace>
    class Channel {
    public:
      using comm_space = CommSpace;

      //template <class CommSpace>
      Channel(){}
      
      // MPI partitioned communication request object
      // int state ; int size ; int side ; int sendparts ;
      // int recvparts ; int readycount ; MPI_Request *request ;
      
    };
    //  TODO: sendinit()
      template<class SendView>
      void sendinit(SendView view) //, KokkosComm::Channel channel)
    {
      Kokkos::Tools::pushRegion("KokkosComm::Impl::send");
      
    }

}  // namespace KokkosComm
