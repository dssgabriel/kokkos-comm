// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <utility>

#include <mpi.h>
#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>
#include <KokkosBlas2_gemv.hpp>

namespace {

namespace kc  = KokkosComm;
namespace kcx = KokkosComm::Experimental;

using Real   = double;
using Matrix = Kokkos::View<Real**>;
using Vector = Kokkos::View<Real*>;
using Scalar = Kokkos::View<Real>;

constexpr auto pipeline_chunk_size(std::size_t extent) -> std::size_t { return (extent + 3) / 4; }

// Logical compute-side traffic per iteration; count one input-vector read per
// GEMV call, including repeated output accumulation and allocated padding.
constexpr auto power_iteration_1d_bytes(std::size_t n, std::size_t local_rows, std::size_t rank) -> double {
  const auto first        = std::min(rank * local_rows, n);
  const auto last         = std::min(first + local_rows, n);
  const auto remote_gemvs = (first != 0) + (last != n);
  const double rows       = static_cast<double>(local_rows);
  // First GEMV (or zero-fill) writes once; each remote GEMV reads and writes.
  // The norm reads once, and normalization reads and writes once.
  return sizeof(Real) * (rows * n + n + rows * (1 + 2 * remote_gemvs + 3));
}

constexpr auto power_iteration_2d_bytes(std::size_t local_rows, std::size_t local_cols, bool normalizes) -> double {
  const auto panel_size  = pipeline_chunk_size(local_cols);
  const auto tile_size   = pipeline_chunk_size(local_rows);
  const auto panels      = (local_cols + panel_size - 1) / panel_size;
  const auto tiles       = (local_rows + tile_size - 1) / tile_size;
  const auto final_panel = (panels - 1) * panel_size;
  const double rows      = static_cast<double>(local_rows);
  // The final input panel is read by each output-tile GEMV separately.
  const double input_reads            = final_panel + static_cast<double>(tiles) * (local_cols - final_panel);
  const double output_accesses        = rows * (2 * panels - 1);
  const double normalization_accesses = normalizes ? 3 * rows : 0;
  return sizeof(Real) * (rows * local_cols + input_reads + output_accesses + normalization_accesses);
}

KOKKOS_INLINE_FUNCTION auto initial_entry(std::size_t global_i) -> Real {
  // Seeded, global-index-based initialization for reproducible random starts across decompositions and rank counts.
  std::uint64_t bits = std::uint64_t{0x123456789abcdef0} + global_i + std::uint64_t{0x9e3779b97f4a7c15};
  bits               = (bits ^ (bits >> 30)) * std::uint64_t{0xbf58476d1ce4e5b9};
  bits               = (bits ^ (bits >> 27)) * std::uint64_t{0x94d049bb133111eb};
  bits ^= bits >> 31;
  return Real{0.5} + static_cast<Real>(bits >> 11) * 0x1.0p-53;
}

#ifdef KOKKOSCOMM_ENABLE_NCCL
// Bootstrap with MPI, using the device already selected by Kokkos. Keep the
// communicator scoped to the benchmark so it is destroyed before finalization.
class NcclContext {
 public:
  NcclContext() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ncclUniqueId id{};
    if (rank == 0) check(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    check(ncclCommInitRank(&comm_, size, id, rank));
  }

  ~NcclContext() { check(ncclCommDestroy(comm_)); }
  NcclContext(const NcclContext&)                    = delete;
  auto operator=(const NcclContext&) -> NcclContext& = delete;
  auto comm() const -> ncclComm_t { return comm_; }

 private:
  static auto check(ncclResult_t result) -> void {
    if (result != ncclSuccess) {
      std::fprintf(stderr, "NCCL: %s\n", ncclGetErrorString(result));
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  ncclComm_t comm_ = nullptr;
};
#endif

// Workspaces are allocated by the caller and reused across all iterations.
template <class CommSpace, class ExecSpace>
auto von_misses_1d_partitioned(
    /// Communicator with its own communication-dedicated execution space instance.
    kc::Communicator<CommSpace, ExecSpace>& comm,
    /// Compute-dedicated execution space instance.
    const ExecSpace& exec,
    /// Rank-local band of A.
    const Matrix& A,
    /// Rank-local block of input vector b_k.
    Vector& b_k,
    /// Global b_k vector with room for equally size blocks from all ranks.
    const Vector& global_b_k,
    /// Rank-local block of output vector b_k1.
    Vector& b_k1,
    /// Squared norm of output vector rank-local output vector b_k1.
    const Scalar& loc_sq_norm,
    /// Squared norm of output vector global output vector b_k1.
    const Scalar& sq_norm,
    /// Number of Von Misses iterations to stop at.
    int niter
) -> void {
  // Clamp column ownership to the real matrix dimension, excluding padding.
  // A rank with only padded rows may also have no locally owned columns.
  const auto first = std::min(comm.rank() * b_k.extent(0), A.extent(1));
  const auto last  = std::min(first + b_k.extent(0), A.extent(1));

  const auto left_A  = Kokkos::subview(A, Kokkos::ALL(), std::make_pair(0ul, first));
  const auto loc_A   = Kokkos::subview(A, Kokkos::ALL(), std::make_pair(first, last));
  const auto right_A = Kokkos::subview(A, Kokkos::ALL(), std::make_pair(last, A.extent(1)));

  const auto left_b_k  = Kokkos::subview(global_b_k, std::make_pair(0ul, first));
  const auto right_b_k = Kokkos::subview(global_b_k, std::make_pair(last, A.extent(1)));

  for (int it = 0; it < niter; ++it) {
    const auto loc_b_k = Kokkos::subview(b_k, std::make_pair(0ul, last - first));
    // b_k was produced on the compute instance. Fencing comm.exec() cannot establish this cross-instance dependency.
    exec.fence("b_k ready for all-gather");

    auto ag_req = kcx::allgather(comm, b_k, global_b_k);

    // At least one real (not padded) locally owned column.
    assert(first < last);
    // Perform the rank-local GEMV while AG is in flight.
    KokkosBlas::gemv(exec, "N", 1.0, loc_A, loc_b_k, 0.0, b_k1);

    ag_req.wait();

    // All b_k1 updates run on exec, so no fence is needed between local, left, and right GEMVs.

    // Compute left-side local GEMV
    // Rank 0 does not have anything to its left so it is excluded
    if (first != 0ul) {
      KokkosBlas::gemv(exec, "N", 1.0, left_A, left_b_k, 1.0, b_k1);
    }
    // Compute right-side local GEMV
    // Rank P-1 does not have anything to its right so it is excluded
    if (last != A.extent(1)) {
      KokkosBlas::gemv(exec, "N", 1.0, right_A, right_b_k, 1.0, b_k1);
    }

    Kokkos::parallel_reduce(
        "local b_k1 squared norm", Kokkos::RangePolicy(exec, 0, b_k.extent(0)),
        KOKKOS_LAMBDA(const int i, Real& sum) { sum += b_k1(i) * b_k1(i); }, loc_sq_norm
    );
    exec.fence("b_k1 loc_sq_norm ready for all-reduce");
    kcx::allreduce(comm, loc_sq_norm, sq_norm, kc::Sum{}).wait();

    Kokkos::parallel_for(
        "normalize b_k1", Kokkos::RangePolicy(exec, 0, b_k.extent(0)),
        KOKKOS_LAMBDA(const int i) {
          const Real norm = Kokkos::sqrt(sq_norm());
          // Safety: guard from division-by-0
          // assert(std::isnormal(norm));
          b_k1(i) = b_k1(i) / norm;
        }
    );
    // View swap for next iteration
    Kokkos::kokkos_swap(b_k, b_k1);
  }
  exec.fence("power iteration complete");
}

// R(i,j) owns A_ij. Column j broadcasts b_k,j from R(j,j); row i
// reduces A_ij*b_k,j to R(i,i), which owns the next vector block.
// With RowAllReduce, completed output blocks are replicated across each row,
// and a scalar all-reduce in each column replaces the diagonal communicator.
template <bool RowAllReduce = false, class CommSpace, class ExecSpace>
auto von_misses_2d_partitioned(
    kc::Communicator<CommSpace, ExecSpace>& row_comm,
    kc::Communicator<CommSpace, ExecSpace>& col_comm,
    kc::Communicator<CommSpace, ExecSpace>* diag_comm,
    const ExecSpace& exec,
    const Matrix& A,
    Vector& b_k,
    const Vector& partial_b_k1,
    Vector& b_k1,
    const Scalar& local_norm_squared,
    const Scalar& norm_squared,
    int iterations
) -> void {
  const int grid_row = col_comm.rank();
  const int grid_col = row_comm.rank();
  const auto rows    = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<std::size_t>>(exec, 0, b_k1.extent(0));
  // Four disjoint input panels and output tiles, with smaller tails when needed.
  const std::size_t panel_size  = pipeline_chunk_size(A.extent(1));
  const std::size_t tile_size   = pipeline_chunk_size(A.extent(0));
  const std::size_t final_panel = ((A.extent(1) - 1) / panel_size) * panel_size;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    // Finish normalization before the next broadcast consumes the iterate.
    exec.fence("2D power iteration input ready for broadcast");
    // Broadcast panel p while GEMV consumes the completed panel p-1. Panels
    // occupy disjoint slices of b_k, including on the in-place broadcast root.
    for (std::size_t first = 0; first < A.extent(1); first += panel_size) {
      auto incoming  = Kokkos::subview(b_k, std::make_pair(first, std::min(first + panel_size, A.extent(1))));
      auto broadcast = kcx::broadcast(col_comm, incoming, grid_col);
      if (first != 0) {
        const auto columns = std::make_pair(first - panel_size, first);
        const auto panel_A = Kokkos::subview(A, Kokkos::ALL(), columns);
        const auto panel_b = Kokkos::subview(b_k, columns);
        KokkosBlas::gemv(exec, "N", Real{1}, panel_A, panel_b, first == panel_size ? Real{0} : Real{1}, partial_b_k1);
      }
      broadcast.wait();
    }

    // Finish the final input panel tile-by-tile. Earlier panel contributions
    // are ordered on exec; each tile is complete before being sent to reduce.
    const auto final_columns = std::make_pair(final_panel, A.extent(1));
    const auto final_b       = Kokkos::subview(b_k, final_columns);
    const auto finish_tile   = [&](std::size_t first) {
      const auto tile_rows    = std::make_pair(first, std::min(first + tile_size, A.extent(0)));
      const auto tile_A       = Kokkos::subview(A, tile_rows, final_columns);
      const auto tile_partial = Kokkos::subview(partial_b_k1, tile_rows);
      KokkosBlas::gemv(exec, "N", Real{1}, tile_A, final_b, final_panel == 0 ? Real{0} : Real{1}, tile_partial);
    };
    finish_tile(0);
    for (std::size_t first = 0; first < A.extent(0); first += tile_size) {
      exec.fence("2D power iteration tile ready for row reduction");
      const auto tile_rows = std::make_pair(first, std::min(first + tile_size, A.extent(0)));
      auto tile_partial    = Kokkos::subview(partial_b_k1, tile_rows);
      auto tile_result     = Kokkos::subview(b_k1, tile_rows);
      auto reduction       = [&]() {
        if constexpr (RowAllReduce) {
          return kcx::allreduce(row_comm, tile_partial, tile_result, kc::Sum{});
        } else {
          return kcx::reduce(row_comm, tile_partial, tile_result, grid_row, kc::Sum{});
        }
      }();
      // The next tile has separate storage from the active reduction buffers.
      if (first + tile_size < A.extent(0)) finish_tile(first + tile_size);
      reduction.wait();
    }

    if (RowAllReduce || diag_comm != nullptr) {
      Kokkos::parallel_reduce(
          "2D power iteration local norm", rows,
          KOKKOS_LAMBDA(const std::size_t i, Real& sum) { sum += b_k1(i) * b_k1(i); }, local_norm_squared
      );
      exec.fence("2D power iteration local norm ready for all-reduce");
      if constexpr (RowAllReduce) {
        // Each column contains exactly one copy of each completed row block.
        kcx::allreduce(col_comm, local_norm_squared, norm_squared, kc::Sum{}).wait();
      } else {
        kcx::allreduce(*diag_comm, local_norm_squared, norm_squared, kc::Sum{}).wait();
      }
      Kokkos::parallel_for(
          "2D power iteration normalize", rows,
          KOKKOS_LAMBDA(const std::size_t i) {
            const Real norm = Kokkos::sqrt(norm_squared());
            // Safety: guard from division-by-0
            // assert(std::isnormal(norm));
            b_k1(i) = b_k1(i) / norm;
          }
      );
    }
    std::swap(b_k, b_k1);
  }
  exec.fence("2D power iteration complete");
}

template <bool TwoDimensional, bool RowAllReduce = false>
auto benchmark_von_misses_partitioned(benchmark::State& state) -> void {
  static_assert(TwoDimensional || !RowAllReduce);
// The MPI backend currently rejects device-view nonblocking all-reduce with
// Open MPI (see issue #215). NCCL does not have that restriction.
#if !defined(KOKKOSCOMM_ENABLE_NCCL) && defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && \
    (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  state.SkipWithError("Power iteration requires device all-reduce, unsupported with Open MPI + CUDA/HIP");
#else
  if (state.range(0) <= 0 || state.range(1) <= 0) {
    state.SkipWithError("Matrix dimension and power iteration count must be positive");
    return;
  }
  using ExecSpace      = Kokkos::DefaultExecutionSpace;
  const auto instances = [] {
    if constexpr (TwoDimensional) {
      // Keep identical compute/communication resource weights for both 2D
      // variants; the fourth instance is unused when there is no diagonal comm.
      return Kokkos::Experimental::partition_space(ExecSpace{}, 3, 1, 1, 1);
    } else {
      return Kokkos::Experimental::partition_space(ExecSpace{}, 1, 1);
    }
  }();
  const auto& exec      = instances[0];
  const auto& comm_exec = instances[1];
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  NcclContext nccl_ctx;
  auto raw_comm         = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto comm             = kc::Communicator<>::from_raw(raw_comm, comm_exec);
  const auto n          = static_cast<std::size_t>(state.range(0));
  const auto iterations = static_cast<int>(state.range(1));
  const auto ranks      = static_cast<std::size_t>(comm.size());
  const int q           = static_cast<int>(std::sqrt(static_cast<double>(ranks)));
  std::optional<decltype(comm)> row_comm, col_comm, diag_comm;
  if constexpr (TwoDimensional) {
    if (q * q != comm.size()) {
      state.SkipWithError("2D power iteration requires a square number of MPI/NCCL ranks");
      return;
    }
    const int grid_row = comm.rank() / q;
    const int grid_col = comm.rank() % q;
    row_comm           = decltype(comm)::split_from_raw(raw_comm, grid_row, grid_col, instances[2]);
    col_comm           = decltype(comm)::split_from_raw(raw_comm, grid_col, grid_row, instances[1]);
    if constexpr (!RowAllReduce) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
      constexpr int no_color = NCCL_SPLIT_NOCOLOR;
#else
      constexpr int no_color = MPI_UNDEFINED;
#endif
      diag_comm = decltype(comm)::split_from_raw(raw_comm, grid_row == grid_col ? 0 : no_color, grid_row, instances[3]);
    }
    // Every rank must either proceed or skip, even if a communicator split fails.
    const int failed = !row_comm || !col_comm || (!RowAllReduce && grid_row == grid_col && !diag_comm);
    int any_failed;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_failed) {
      state.SkipWithError("Failed to create 2D process-grid communicators");
      return;
    }
  }
  const std::size_t row_blocks = TwoDimensional ? q : ranks;
  const std::size_t local_m    = (n + row_blocks - 1) / row_blocks;
  const std::size_t local_n    = TwoDimensional ? local_m : n;
  const std::size_t row_offset = (TwoDimensional ? comm.rank() / q : comm.rank()) * local_m;
  const std::size_t col_offset = TwoDimensional ? (comm.rank() % q) * local_n : 0;

  Matrix A(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "A"), local_m, local_n);
  Vector b_k(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "b_k"), local_m);
  Vector global_b_k;
  Vector partial_b_k1;
  if constexpr (TwoDimensional) {
    partial_b_k1 = Vector(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "partial b_k1"), local_m);
  } else {
    global_b_k = Vector(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "gathered b_k"), local_m * ranks);
  }
  Vector b_k1(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "b_k1"), local_m);
  Scalar local_norm_squared("local norm squared");
  Scalar norm_squared("norm squared");

  const double local_algorithm_bytes =
      TwoDimensional ? power_iteration_2d_bytes(local_m, local_n, RowAllReduce || diag_comm.has_value())
                     : power_iteration_1d_bytes(n, local_m, comm.rank());
  double algorithm_bytes;
  MPI_Allreduce(&local_algorithm_bytes, &algorithm_bytes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // A = I + u*u^T / (u^T*u), u_i = 1 + i/n. Its dominant eigenvalue is 2,
  // with eigenvector u/||u||; every other eigenvalue is 1.
  const Real rn             = static_cast<Real>(n);
  const Real u_norm_squared = rn + (rn - 1) + (rn - 1) * (2 * rn - 1) / (6 * rn);
  Kokkos::parallel_for(
      "initialize power iteration matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>, Kokkos::IndexType<std::size_t>>(
          exec, {0, 0}, {local_m, local_n}
      ),
      KOKKOS_LAMBDA(const std::size_t i, const std::size_t j) {
        const std::size_t global_i = row_offset + i;
        const std::size_t global_j = col_offset + j;
        A(i, j)                    = global_i < n && global_j < n ? (global_i == global_j ? Real{1} : Real{0}) +
                                                     (1 + global_i / rn) * (1 + global_j / rn) / u_norm_squared
                                                                  : Real{0};
      }
  );

  const auto rows = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<std::size_t>>(exec, 0, local_m);
  Vector initial_b(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "initial b"), local_m);
  Kokkos::parallel_for(
      "initialize power iteration vector", rows,
      KOKKOS_LAMBDA(const std::size_t i) {
        initial_b(i) = row_offset + i < n ? initial_entry(row_offset + i) : Real{0};
      }
  );
  Real local_initial_norm_squared = 0;
  // Count one owner per row block, including when no diagonal communicator exists.
  if (!TwoDimensional || comm.rank() / q == comm.rank() % q) {
    Kokkos::parallel_reduce(
        "initial vector local norm", rows,
        KOKKOS_LAMBDA(const std::size_t i, Real& sum) { sum += initial_b(i) * initial_b(i); },
        local_initial_norm_squared
    );
  }
  Real initial_norm_squared;
  MPI_Allreduce(&local_initial_norm_squared, &initial_norm_squared, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const Real initial_scale = 1 / std::sqrt(initial_norm_squared);
  Kokkos::parallel_for(
      "normalize initial vector", rows, KOKKOS_LAMBDA(const std::size_t i) { initial_b(i) *= initial_scale; }
  );
  auto reset = [&]() {
    Kokkos::deep_copy(exec, b_k, initial_b);
    exec.fence("power iteration setup complete");
  };
  auto iterate = [&]() {
    if constexpr (TwoDimensional) {
      von_misses_2d_partitioned<RowAllReduce>(
          *row_comm, *col_comm, diag_comm ? &*diag_comm : nullptr, exec, A, b_k, partial_b_k1, b_k1, local_norm_squared,
          norm_squared, iterations
      );
    } else {
      von_misses_1d_partitioned(comm, exec, A, b_k, global_b_k, b_k1, local_norm_squared, norm_squared, iterations);
    }
  };

  // Warm up kernels and communication before collecting measurements.
  reset();
  iterate();

  for (auto _ : state) {
    reset();
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    iterate();
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed;
    MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // Identical manual times keep Google Benchmark's adaptive iteration count
    // in sync across ranks. Setup, reset and timing reductions are excluded.
    state.SetIterationTime(max_elapsed);
  }

  state.counters["ranks"]            = static_cast<double>(ranks);
  state.counters["power_iterations"] = iterations;
  Real eigenvalue                    = 0;
  if (comm.rank() == 0) {
    const auto host_norm_squared = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, norm_squared);
    eigenvalue                   = std::sqrt(host_norm_squared());
  }
  MPI_Bcast(&eigenvalue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // ||A*b_k|| approaches the dominant eigenvalue for this positive-definite A.
  state.counters["eigenvalue"] = eigenvalue;
  state.counters["matvec_flops"] =
      benchmark::Counter(2 * rn * rn * iterations, benchmark::Counter::kIsIterationInvariantRate);
  // Aggregate logical matrix/vector GB/s; communication and scalar bookkeeping are excluded.
  state.counters["algobw"] =
      benchmark::Counter(algorithm_bytes * iterations / 1e9, benchmark::Counter::kIsIterationInvariantRate);
  state.SetItemsProcessed(state.iterations() * iterations);
#endif
}

// HBM-resident doubles, one rank per GH200, four GPUs per node. Budget
// at most 72 GiB of matrix storage per GPU on the 96 GB HBM configuration,
// leaving room for vectors, communication buffers and runtime allocations.
void power_iteration_sizes(benchmark::Benchmark* benchmark) {
  constexpr int iterations = 50;
  for (const auto n : {4096, 8192, 16384, 32768, 65536, 131072, 196608}) {
    benchmark->Args({n, iterations});
  }
  // Aggregate matrix storage: 512, 800 and 1152 GiB. Under this budget,
  // 1D needs 2, 3 and 4 fully populated nodes (8, 12 and 16 ranks).
  // 2D requires square grids: use 9 ranks for 262144, or 16 for any size.
#ifdef KOKKOSCOMM_BENCHMARKS_ENABLE_LARGE_SCALE
  for (const auto n : {262144, 327680, 393216}) {
    benchmark->Args({n, iterations});
  }
#endif
}

BENCHMARK_TEMPLATE(benchmark_von_misses_partitioned, false)
    ->Name("von_misses/1d")
    ->ArgNames({"N", "power_iterations"})
    ->Apply(power_iteration_sizes)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(benchmark_von_misses_partitioned, true)
    ->Name("von_misses/2d")
    ->ArgNames({"N", "power_iterations"})
    ->Apply(power_iteration_sizes)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(benchmark_von_misses_partitioned, true, true)
    ->Name("von_misses/2d_allreduce")
    ->ArgNames({"N", "power_iterations"})
    ->Apply(power_iteration_sizes)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace
