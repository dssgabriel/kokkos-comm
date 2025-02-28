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

  template <class CommSpace>
  class Channel {
  public:
    using comm_space = CommSpace;
    
    Channel() : state_(0), send_parts_(0), recv_parts_(0), ready_count_(0) {
      // Initialize MPI requests
      requests_ = new MPI_Request[2]; // Example: 2 requests for send/recv
    }

    Channel(int dest_rank, int src_rank, int tag, MPI_Comm comm)
    : dest_rank_(dest_rank), 
      src_rank_(src_rank), 
      tag_(tag), 
      comm_(comm),
      state_(0),
      send_parts_(0),
      recv_parts_(0),
      ready_count_(0) {
        requests_ = new MPI_Request[2];
    }

    ~Channel() {
      delete[] requests_; // TODO: Only free if in inactive state
    }
    
    template <class SendView>
    void sendinit(SendView view) {
      Kokkos::Tools::pushRegion("KokkosComm::Channel::sendinit");
      using value_type = typename SendView::value_type;
      // Initialize persistent send
      MPI_Send_init(view.data(), view.size(), KokkosComm::Impl::mpi_type_v<value_type>, dest_rank_, tag_, comm_, &requests_[0]);
      Kokkos::Tools::popRegion();
    }
    
    template <class RecvView>
    void recvinit(RecvView view) {
      Kokkos::Tools::pushRegion("KokkosComm::Channel::recvinit");
      using value_type = typename RecvView::value_type;
      // Initialize persistent receive
      MPI_Recv_init(view.data(), view.size(), KokkosComm::Impl::mpi_type_v<value_type>, src_rank_, tag_, comm_, &requests_[1]);
      Kokkos::Tools::popRegion();
    }
    
    void start() {
      Kokkos::Tools::pushRegion("KokkosComm::Channel::start");
      MPI_Startall(2, requests_); // Start all requests
      Kokkos::Tools::popRegion();
      // TODO: When completed, destroy OR maintain Channel
    }

    void wait() {
      Kokkos::Tools::pushRegion("KokkosComm::Channel::wait");
      MPI_Waitall(2, requests_, MPI_STATUSES_IGNORE); // Wait for all requests
      Kokkos::Tools::popRegion();
    }

  private:
    int state_;             // Communication state
    int send_parts_;        // # of send partitions
    int recv_parts_;        // # of receive partitions
    int ready_count_;       // # of ready partitions
    MPI_Request* requests_; // MPI requests for send/recv
    int dest_rank_;         // Destination rank for send
    int src_rank_;          // Source rank for receive
    int tag_;               // MPI tag
    MPI_Comm comm_;         // MPI communicator
  };

}  // namespace KokkosComm
