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

#include <unistd.h>  // gethostname

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iosfwd>
#include <string>
#include <string_view>

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#pragma once

namespace {

#define MPI_CHECK(cmd)                                                                                     \
  do {                                                                                                     \
    if (int res = cmd; res != MPI_SUCCESS) {                                                               \
      std::cerr << std::format("KokkosComm::unit_tests: {}:{} error(MPI): {}\n", __FILE__, __LINE__, res); \
      Kokkos::abort();                                                                                     \
    }                                                                                                      \
  } while (0)

#define NCCL_CHECK(cmd)                                                                               \
  do {                                                                                                \
    ncclResult_t r = cmd;                                                                             \
    if (ncclResult_t res; res != ncclSuccess) {                                                       \
      std::cerr << std::format("KokkosComm::unit_tests: {}:{} error(NCCL): {}\n", __FILE__, __LINE__, \
                               ncclGetErrorString(res));                                              \
      Kokkos::abort();                                                                                \
    }                                                                                                 \
  } while (0)

#define CUDA_CHECK(cmd)                                                                               \
  do {                                                                                                \
    cudaError_t e = cmd;                                                                              \
    if (cudaError_t res = cmd; res != cudaSuccess) {                                                  \
      std::cerr << std::format("KokkosComm::unit_tests: {}:{} error(CUDA): {}\n", __FILE__, __LINE__, \
                               cudaGetErrorString(res));                                              \
      Kokkos::abort();                                                                                \
    }                                                                                                 \
  } while (0)

constexpr std::string_view HOSTID_FILE = "/proc/sys/kernel/random/boot_id";

[[nodiscard]] constexpr auto get_hash(std::string_view str) noexcept -> uint64_t {
  uint64_t result = 5381;
  for (unsigned char c : str) result = ((result << 5) + result) ^ c;  // result * 33 ^ c
  return result;
}

/// Generate a hash of the unique identifying string for this host that will be unique for both bare-metal and
/// container instances.
/// Equivalent of a hash of:
/// ```sh
/// $(hostname)$(cat /proc/sys/kernel/random/boot_id)
/// ```
[[nodiscard]] auto hash_hostname(std::string_view hostname) -> uint64_t {
  std::string combined{hostname};
  if (std::ifstream file{std::string{HOSTID_FILE}}; file) {
    std::string boot_id;
    if (file >> boot_id) combined += boot_id;
  }
  return get_hash(combined);
}

[[nodiscard]] auto get_hostname() -> std::string {
  std::array<char, 256> buf{};
  if (::gethostname(buf.data(), buf.size()) != 0) {
    return "hostname";
  }
  std::string hostname{buf.data()};
  if (auto dot_pos = hostname.find('.'); dot_pos != std::string::npos) {
    hostname.resize(dot_pos);
  }
  return hostname;
}

}  // namespace

namespace test_utils::nccl {

class Ctx {
 public:
  static auto init() -> Ctx {
    // Setup MPI Session and create communicator
    MPI_Session mpi_session = MPI_SESSION_NULL;
    MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &mpi_session);
    MPI_Group mpi_group = MPI_GROUP_NULL;
    MPI_Group_from_session_pset(mpi_session, "mpi://WORLD", &mpi_group);
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    MPI_Comm_create_from_group(mpi_group, "kokkos-comm.test.mpi-comm", MPI_INFO_NULL, MPI_ERRORS_RETURN, &mpi_comm);

    // Retrieve rank info
    int n_ranks, my_rank;
    MPI_Comm_size(comm, &n_ranks);
    MPI_Comm_rank(comm, &my_rank);

    // Compute `local_rank` based on hostname which is used in selecting a GPU
    int local_rank;
    std::vector<uint64_t> hostname_hashes(n_pe);
    auto hostname       = get_hostname();
    host_hashs[my_rank] = hash_hostname(hostname);
    MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostname_hashes.data(), sizeof(uint64_t), MPI_BYTE,
                            mpi_comm));
    for (int p = 0; p < n_ranks; ++p) {
      if (host_hashes[p] == host_hashes[my_rank]) {
        local_rank++;
      }
    }
    CUDA_CHECK(cudaSetDevice(local_rank));

    // Get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId_t nccl_id;
    if (my_rank == 0) {
      ncclGetUniqueId(&nccl_id);
    }
    MPI_CHECK(MPI_Bcast(static_cast<void *>(&nccl_id), sizeof(nccl_id), MPI_BYTE, 0, mpi_comm));

    // Don't need MPI anymore past this point
    MPI_Comm_free(&mpi_comm);
    MPI_Group_free(&mpi_group);
    MPI_Session_finalize(&mpi_session);

    // NCCL comm configuration
    ncclConfig_t nccl_cfg = NCCL_CONFIG_INITIALIZER;
    nccl_cfg.blocking     = 0;
    nccl_cfg.commName     = "kokkos-comm.test.nccl-comm";
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRankConfig(&nccl_comm, n_ranks, id, my_rank, &nccl_cfg));

    return Ctx(nccl_comm, n_ranks, my_rank);
  }

  auto comm() -> ncclComm_t & { return comm_ }
  auto n_ranks() -> int { return n_ranks_; }
  auto my_rank() -> int { return my_rank_; }

 private:
  explicit Ctx(ncclComm_t comm, int n_ranks, int my_rank) : comm_(comm), n_ranks_(n_ranks), my_rank_(my_rank) {}
  ~Ctx() { NCCL_CHECK(ncclCommDestroy(comm_)); }

  Ctx(const Ctx &)                     = delete;
  auto operator=(const Ctx &) -> Ctx & = delete;
  Ctx(Ctx &&)                          = delete;
  auto operator=(Ctx &&) -> Ctx &      = delete;
  ncclComm_t comm_;
  int n_ranks_;
  int my_rank_;
};

}  // namespace test_utils::nccl
