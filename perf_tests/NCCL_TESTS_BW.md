# Compare the bandwidth benchmarks with nccl-tests

Use the same CUDA, NCCL, MPI, rank placement, and NCCL environment variables as the Kokkos Comm run.

## Build nccl-tests

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
git rev-parse HEAD

make -j8 \
  MPI=1 \
  NAME_SUFFIX=_mpi \
  CUDA_HOME=/path/to/cuda \
  NCCL_HOME=/path/to/nccl \
  MPI_HOME=/path/to/mpi
```

## Run

Run one MPI process per GPU. Set `NRANKS` to the total GPU count.

```bash
NRANKS=8

ARGS=(
  -b 1K -e 256M -f 2
  -d float
  -w 1 -n 20 -m 1
  -z 0 -a 3
  -t 1 -g 1
  -N 1 -c 1
  -D 0 -R 0 -G 0
)

mpirun -np "$NRANKS" ./build/all_reduce_perf_mpi "${ARGS[@]}" -o sum
mpirun -np "$NRANKS" ./build/all_gather_perf_mpi "${ARGS[@]}"
mpirun -np "$NRANKS" ./build/alltoall_perf_mpi "${ARGS[@]}"
```

For a multi-node run, add the launcher option that places one rank on each GPU, such as `-N <gpus-per-node>` or MPICH/Hydra's `-ppn <gpus-per-node>`.

## Compare

- Compare only the **out-of-place** `time`, `algbw`, and `busbw` columns. Ignore the in-place columns.
- Match nccl-tests' `size` column to the Kokkos Comm benchmark's `bytes` counter.
- Use `algbw` as the primary comparison. `busbw` is a rank-normalized value derived from it.
- Check that both executables resolve the same NCCL library before comparing results:

```bash
ldd ./build/all_reduce_perf_mpi | rg 'libnccl|libmpi|libcuda'
ldd /path/to/bench.core.allreduce_bw | rg 'libnccl|libmpi|libcuda'
```
