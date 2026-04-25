#include <memory>
#include <mutex>

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include "../logging.hpp"

#include "utils.hpp"

namespace {

[[nodiscard]] auto get_local_rank(MPI_Comm comm, int my_rank) -> int {
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &node_comm);

  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);

  MPI_Comm_free(&node_comm);
  return node_rank;
}

}  // namespace

namespace test_utils::nccl {

std::unique_ptr<Ctx> Ctx::instance_{};
std::once_flag Ctx::init_flag_{};

Ctx::Ctx(ncclComm_t comm, cudaStream_t stream, int dev, int n_ranks, int my_rank)
    : comm_(comm), stream_(stream), dev_(dev), n_ranks_(n_ranks), my_rank_(my_rank) {}

Ctx::~Ctx() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
  if (comm_ != nullptr) {
    ncclCommDestroy(comm_);
  }
}

auto Ctx::init(bool verbose) -> void {
  std::call_once(init_flag_, [verbose]() {
    int flag = 0;
    KC_MPI_CHECK(MPI_Initialized(&flag));
    KC_CHECK(flag != 0, "MPI is not initialized");

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int n_ranks = 0;
    int my_rank = 0;
    KC_MPI_CHECK(MPI_Comm_size(mpi_comm, &n_ranks));
    KC_MPI_CHECK(MPI_Comm_rank(mpi_comm, &my_rank));

    int local_rank = get_local_rank(mpi_comm, my_rank);

    int n_gpus = 0;
    KC_CUDA_CHECK(cudaGetDeviceCount(&n_gpus));

    if (verbose) {
      KC_INFO("P{} found {} CUDA devices", my_rank, n_gpus);
    }

    KC_CHECK(local_rank < n_gpus, "P{} needs device #{} but only {} devices available", my_rank, local_rank, n_gpus);

    KC_CUDA_CHECK(cudaSetDevice(local_rank));

    if (verbose) {
      KC_INFO("P{} assigned to CUDA device #{}", my_rank, local_rank);
    }

    ncclUniqueId nccl_id{};
    if (my_rank == 0) {
      KC_NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }

    KC_MPI_CHECK(MPI_Bcast(&nccl_id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, mpi_comm));

    ncclComm_t nccl_comm = nullptr;
    KC_NCCL_CHECK(ncclCommInitRank(&nccl_comm, n_ranks, nccl_id, my_rank));

    cudaStream_t stream = nullptr;
    KC_CUDA_CHECK(cudaStreamCreate(&stream));

    instance_ = std::unique_ptr<Ctx>(new Ctx(nccl_comm, stream, local_rank, n_ranks, my_rank));
  });
}

auto Ctx::fini() -> void { instance_.reset(); }

auto Ctx::get() -> Ctx& {
  KC_CHECK(instance_ != nullptr, "NCCL context not initialized");
  return *instance_;
}

auto Ctx::comm() const -> ncclComm_t { return comm_; }

auto Ctx::stream() const -> cudaStream_t { return stream_; }

auto Ctx::size() const -> int { return n_ranks_; }

auto Ctx::rank() const -> int { return my_rank_; }

auto Ctx::device() const -> int { return dev_; }

}  // namespace test_utils::nccl
