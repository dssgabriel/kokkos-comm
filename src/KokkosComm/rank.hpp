// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

struct Rank {
  explicit constexpr Rank(int val) : value(val) {}
  int value;
}

constexpr auto
operator""_rank(unsigned long long val) -> Rank {
  return Rank(static_cast<int>(val));
}
