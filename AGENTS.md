# Kokkos Comm Agent Instructions

This file provides guidelines for agents working in the Kokkos Comm codebase.

## Project Overview

Kokkos Comm is an experimental performance portable explicit communication interface for the Kokkos ecosystem.
- **Backends**: MPI (primary), NCCL (experimental)
- **Language**: C++20 (required)
- **Dependencies**: Kokkos 4.7+, MPI

## Build System

### Source development environment

This project uses Spack environments for development.

The available environments, denoted as `SPACK_ENV` hereafter, are:
- `kc-ompi-omp`: Kokkos w/ OpenMP backend and Open MPI
- `kc-mpich-omp`: Kokkos w/ OpenMP backend and MPICH
- `kc-ompi-cuda`: Kokkos w/ CUDA backend and Open MPI
- `kc-mpich-cuda`: Kokkos w/ CUDA backend and MPICH
<!-- - `kc-nccl-cuda`: Kokkos w/ CUDA backend and NCCL -->


```bash
# Source Spack
source $HOME/oss/spack/share/spack/setup-env.sh

# Activate a Spack environment
spack env activate <SPACK_ENV>
```

### Configure and build

```bash
# Configure with tests enabled
cmake \
  -B build/<SPACK_ENV> \
  -G Ninja \
  -DKokkosComm_ENABLE_MPI=ON \
  -DKokkosComm_ENABLE_TESTS=ON \
  -DKokkosComm_ENABLE_PERFTESTS=ON

# Build
cmake --build build/<SPACK_ENV>
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `KokkosComm_ENABLE_TESTS` | OFF | Build unit tests |
| `KokkosComm_ENABLE_PERFTESTS` | OFF | Build performance tests |
| `KokkosComm_ENABLE_MPI` | ON | Enable MPI backend |
| `KokkosComm_ENABLE_NCCL` | OFF | Enable NCCL backend |
| `KokkosComm_ABORT_ON_ERROR` | OFF | Global abort on runtime errors |

### Dependencies

- Kokkos 4.7+ (required)
- MPI (required unless NCCL only)
- NCCL 2+ (if `KokkosComm_ENABLE_NCCL=ON`)
- GoogleTest (fetched automatically for tests)
- fmt (fetched automatically for tests)

## Testing

### Run all tests

```bash
# Run tests
ctest --test-dir build/<SPACK_ENV>
```

### Run a single test suite

```bash
# Run a specific test suite
ctest --test-dir build/<SPACK_ENV> -R test.core.p2p
```

### Manually run a single test

```bash
# MPI tests require mpirun with correct process count
mpirun -np <N> ./path/to/test_executable

# Example: run a specific test requiring 2 processes
mpirun -np 2 ./unit_tests/test.core.p2p
```

### Test Naming Convention

Tests follow the pattern: `<category>.<backend>.<name>`

- `test.core.*` - Core API tests (require KokkosComm)
- `test.mpi.*` - MPI backend tests
- `test.nccl.*` - NCCL backend tests
- `smoke.*` - Smoke tests (standalone, no KokkosComm)

### Test File Organization

- `unit_tests/test_main.cpp` - GoogleTest main with MPI initialization
- `unit_tests/mpi/` - MPI-specific tests
- `unit_tests/nccl/` - NCCL-specific tests
- `unit_tests/*.cpp` - Core API tests

## Linting and Formatting

### Run Pre-commit Checks

```bash
pre-commit run --all-files
```

### Run Individual Linters

```bash
# C++ formatting (clang-format)
clang-format --style=file -i <files>

# CMake formatting (gersemi)
gersemi --config=.cmake-format <files>
```

### Pre-commit Configuration

See `.pre-commit-config.yaml`:
- `clang-format` - C++ formatting (Google style, 120 char limit)
- `gersemi` - CMake formatting
- `typos` - Spell checking
- Standard hooks (trailing whitespace, EOF fixer, etc.)

## Code Style Guidelines

### File Headers

Every source file must include:

```cpp
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once
// ... rest of file
```

### C++ Style (Google-based, 120 char limit)

Based on `.clang-format`:
- Column limit: 120
- No includes sorting (already handled by pre-commit)
- `BasedOnStyle: google`

Key conventions:
- Use `auto` for type deduction when type is clear
- Prefer `const` references for function parameters: `const Type&`
- Use `noexcept` where applicable
- Use `[[nodiscard]]` for functions returning values
- Use `if constexpr` for compile-time conditionals
- Lambda style: `[=, &var](...)` prefer pass-by-value for captures

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Types/Classes | PascalCase | `ContiguousView` |
| Functions | snake_case | `send_data` |
| Variables | snake_case | `view_data` |
| Constants | kCamelCase | `kMaxSize` |
| Template params | PascalCase | `typename View` |
| Concepts | PascalCase | `KokkosView` |
| Namespaces | snake_case | `KokkosComm::mpi` |

### Namespace Conventions

```cpp
namespace KokkosComm {           // Public API
namespace KokkosComm::Impl {     // Implementation details
namespace KokkosComm::mpi {      // Backend-specific
namespace KokkosComm::Experimental {
```

### Include Order

1. C system headers (`<unistd.h>`, `<cassert>`)
1. C++ system headers (`<type_traits>`, `<utility>`)
1. External library headers (`<mpi.h>`, `<Kokkos_Core.hpp>`)
2. Project headers (`<KokkosComm/concepts.hpp>`)
3. Local relative includes (`"impl/foo.hpp"`)

### Concepts and Constraints

Use C++20 concepts for type constraints:
```cpp
template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <KokkosView View>
void process(View& v) { ... }
```

### Error Handling

Use `Kokkos::abort()` for unrecoverable errors:
```cpp
Kokkos::abort("error message");
```

For MPI errors, use the `fail_if` helper:
```cpp
KokkosComm::mpi::fail_if(mpi_error_code, "descriptive message");
```

### Kokkos-Specific Patterns

- Use `Kokkos::Tools::pushRegion/popRegion` for profiling
- Use `KokkosComm::Impl::allocate_contiguous_for` for packing non-contiguous views
- Use traits like `KokkosComm::span()` instead of `view.span()` for portability
- Fence execution spaces before MPI calls: `space.fence("description")`
- Use `KOKKOS_LAMBDA` macro for portability

### Documentation

Use Doxygen-style comments for public APIs:
```cpp
/// @brief Brief description
/// @tparam T Description of template parameter
/// @param v Description of parameter
/// @returns Description of return value
template <typename T>
auto process(T v) -> Result;
```

## Project Structure

```
src/KokkosComm/
├── KokkosComm.hpp          # Main include file
├── concepts.hpp            # C++20 concepts
├── traits.hpp              # View traits and helpers
├── point_to_point.hpp      # Send/recv APIs
├── collective.hpp          # Collective operations
├── datatype.hpp            # MPI datatype conversions
├── reduction_op.hpp        # Reduction operators
├── impl/
│   └── contiguous.hpp      # Contiguous view utilities
├── mpi/                    # MPI backend
│   ├── send.hpp, recv.hpp
│   ├── allreduce.hpp, reduce.hpp
│   ├── broadcast.hpp, barrier.hpp
│   ├── alltoall.hpp, allgather.hpp
│   └── impl/               # Internal MPI implementation
└── nccl/                   # NCCL backend (experimental)
    └── ... (same structure)
```

## Common Tasks

### Adding a New Test

1. Create test file in `unit_tests/mpi/` or `unit_tests/nccl/`
2. Use GoogleTest patterns (TEST, TYPED_TEST, etc.)
3. Add to `unit_tests/CMakeLists.txt` using `kc_add_unit_test`
4. Configure with `NUM_PES N` for MPI process count

### Adding a New Backend Function

1. Add declaration to appropriate header (`collective.hpp`, `point_to_point.hpp`)
2. Implement in backend-specific directory (`mpi/`, `nccl/`)
3. Use existing patterns (packer, request, handle)
4. Add tests

### Modifying CMakeLists.txt

- Follow gersemi formatting (run `gersemi` to format)
- Use 2-space indentation
- Line length: 120 characters
