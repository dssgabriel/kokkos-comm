// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <KokkosComm/concepts.hpp>

namespace KokkosComm::Experimental {

struct Nccl {};

}  // namespace KokkosComm::Experimental

// Nccl is a KokkosComm::CommunicationSpace
template <>
struct KokkosComm::Impl::is_communication_space<KokkosComm::Experimental::Nccl> : public std::true_type {};
