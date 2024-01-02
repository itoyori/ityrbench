/*
 * Original code from the Cilk project
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/*
 * this program uses an algorithm that we call `cilksort'.
 * The algorithm is essentially mergesort:
 *
 *   cilksort(in[1..n]) =
 *       spawn cilksort(in[1..n/2], tmp[1..n/2])
 *       spawn cilksort(in[n/2..n], tmp[n/2..n])
 *       sync
 *       spawn cilkmerge(tmp[1..n/2], tmp[n/2..n], in[1..n])
 *
 *
 * The procedure cilkmerge does the following:
 *
 *       cilkmerge(A[1..n], B[1..m], C[1..(n+m)]) =
 *          find the median of A \union B using binary
 *          search.  The binary search gives a pair
 *          (ma, mb) such that ma + mb = (n + m)/2
 *          and all elements in A[1..ma] are smaller than
 *          B[mb..m], and all the B[1..mb] are smaller
 *          than all elements in A[ma..n].
 *
 *          spawn cilkmerge(A[1..ma], B[1..mb], C[1..(n+m)/2])
 *          spawn cilkmerge(A[ma..m], B[mb..n], C[(n+m)/2 .. (n+m)])
 *          sync
 *
 * The algorithm appears for the first time (AFAIK) in S. G. Akl and
 * N. Santoro, "Optimal Parallel Merging and Sorting Without Memory
 * Conflicts", IEEE Trans. Comp., Vol. C-36 No. 11, Nov. 1987 .  The
 * paper does not express the algorithm using recursion, but the
 * idea of finding the median is there.
 *
 * For cilksort of n elements, T_1 = O(n log n) and
 * T_\infty = O(log^3 n).  There is a way to shave a
 * log factor in the critical path (left as homework).
 */

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <ctime>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <algorithm>
#include <type_traits>
#include <limits>
#include <random>

#include "mtbb/task_group.h"
#include "mtbb/parallel_for.h"
#include "pcg_random.hpp"

template <typename T> constexpr inline const char* typename_str()         { return "unknown"; }
template <>           constexpr inline const char* typename_str<char>()   { return "char";    }
template <>           constexpr inline const char* typename_str<short>()  { return "short";   }
template <>           constexpr inline const char* typename_str<int>()    { return "int";     }
template <>           constexpr inline const char* typename_str<long>()   { return "long";    }
template <>           constexpr inline const char* typename_str<float>()  { return "float";   }
template <>           constexpr inline const char* typename_str<double>() { return "double";  }

inline uint64_t clock_gettime_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000 + (uint64_t)ts.tv_nsec;
}

template <typename T>
class span {
  using this_t = span<T>;

public:
  using element_type = T;
  using value_type   = std::remove_cv_t<T>;
  using size_type    = std::size_t;
  using pointer      = T*;
  using iterator     = pointer;
  using reference    = T&;

  span() {}
  template <typename ContiguousIterator>
  span(ContiguousIterator first, size_type n)
    : ptr_(&(*first)), n_(n) {}
  template <typename ContiguousIterator>
  span(ContiguousIterator first, ContiguousIterator last)
    : ptr_(&(*first)), n_(last - first) {}
  template <typename U>
  span(span<U> s) : ptr_(s.data()), n_(s.size() * sizeof(U) / sizeof(T)) {}

  constexpr pointer data() const noexcept { return ptr_; }
  constexpr size_type size() const noexcept { return n_; }

  constexpr iterator begin() const noexcept { return ptr_; }
  constexpr iterator end() const noexcept { return ptr_ + n_; }

  constexpr reference operator[](size_type i) const { assert(i <= n_); return ptr_[i]; }

  constexpr reference front() const { return *ptr_; }
  constexpr reference back() const { return *(ptr_ + n_ - 1); }

  constexpr bool empty() const noexcept { return n_ == 0; }

  constexpr this_t subspan(size_type offset, size_type count) const {
    assert(offset + count <= n_);
    return {ptr_ + offset, count};
  }

private:
  pointer   ptr_ = nullptr;
  size_type n_   = 0;
};

#ifndef ITYRBENCH_ELEM_TYPE
#define ITYRBENCH_ELEM_TYPE int
#endif
using elem_t = ITYRBENCH_ELEM_TYPE;

std::size_t n_input       = std::size_t(1) * 1024 * 1024;
int         n_repeats     = 10;
std::size_t cutoff_sort   = std::size_t(4) * 1024;
std::size_t cutoff_merge  = std::size_t(4) * 1024;
bool        verify_result = true;

template <typename T>
auto divide(const span<T>& s, typename span<T>::size_type at) {
  return std::make_pair(s.subspan(0, at), s.subspan(at, s.size() - at));
}

template <typename T>
auto divide_two(const span<T>& s) {
  return divide(s, s.size() / 2);
}

template <typename T>
std::size_t binary_search(span<T> s, const T& v) {
  auto it = std::lower_bound(s.begin(), s.end(), v);
  return it - s.begin();
}

template <typename T>
void cilkmerge(span<T> s1,
               span<T> s2,
               span<T> dest) {
  assert(s1.size() + s2.size() == dest.size());

  if (s1.size() < s2.size()) {
    // s2 is always smaller
    std::swap(s1, s2);
  }

  if (s2.size() == 0) {
    std::copy(s1.begin(), s1.end(), dest.begin());
    return;
  }

  if (dest.size() < cutoff_merge) {
    std::merge(s1.begin(), s1.end(), s2.begin(), s2.end(), dest.begin());
    return;
  }

  std::size_t split1 = (s1.size() + 1) / 2;
  std::size_t split2 = binary_search(s2, s1[split1 - 1]);

  auto [s11  , s12  ] = divide(s1, split1);
  auto [s21  , s22  ] = divide(s2, split2);
  auto [dest1, dest2] = divide(dest, split1 + split2);

  mtbb::task_group tg;
  /* tg.run([&]{ cilkmerge(s11, s21, dest1); }); */
  /* tg.run([&]{ cilkmerge(s12, s22, dest2); }); */
  tg.run([&s11 = s11, &s21 = s21, &dest1 = dest1]{ cilkmerge(s11, s21, dest1); });
  tg.run([&s12 = s12, &s22 = s22, &dest2 = dest2]{ cilkmerge(s12, s22, dest2); });
  tg.wait();
}

template <typename T>
void cilksort(span<T> a, span<T> b) {
  assert(a.size() == b.size());

  if (a.size() < cutoff_sort) {
    std::sort(a.begin(), a.end());
    return;
  }

  auto [a12, a34] = divide_two(a);
  auto [b12, b34] = divide_two(b);

  auto [a1, a2] = divide_two(a12);
  auto [a3, a4] = divide_two(a34);
  auto [b1, b2] = divide_two(b12);
  auto [b3, b4] = divide_two(b34);

  mtbb::task_group tg;
  /* tg.run([&]{ cilksort(a1, b1); }); */
  /* tg.run([&]{ cilksort(a2, b2); }); */
  /* tg.run([&]{ cilksort(a3, b3); }); */
  /* tg.run([&]{ cilksort(a4, b4); }); */
  tg.run([&a1 = a1, &b1 = b1]{ cilksort(a1, b1); });
  tg.run([&a2 = a2, &b2 = b2]{ cilksort(a2, b2); });
  tg.run([&a3 = a3, &b3 = b3]{ cilksort(a3, b3); });
  tg.run([&a4 = a4, &b4 = b4]{ cilksort(a4, b4); });
  tg.wait();

  /* tg.run([&]{ cilkmerge(a1, a2, b12); }); */
  /* tg.run([&]{ cilkmerge(a3, a4, b34); }); */
  tg.run([&a1 = a1, &a2 = a2, &b12 = b12]{ cilkmerge(a1, a2, b12); });
  tg.run([&a3 = a3, &a4 = a4, &b34 = b34]{ cilkmerge(a3, a4, b34); });
  tg.wait();

  cilkmerge(b12, b34, a);
}

template <typename T, typename Rng>
std::enable_if_t<std::is_integral_v<T>, T>
gen_random_elem(Rng& r) {
 static std::uniform_int_distribution<T> dist(0, std::numeric_limits<T>::max());
 return dist(r);
}

template <typename T, typename Rng>
std::enable_if_t<std::is_floating_point_v<T>, T>
gen_random_elem(Rng& r) {
  static std::uniform_real_distribution<T> dist(0, 1.0);
  return dist(r);
}

template <typename T>
void fill_array(span<T> s) {
  static int counter = 0;
  auto seed = counter++;

  mtbb::parallel_for_grainsize_aux(
      std::size_t(0), std::size_t(0), s.size(), std::size_t(1), cutoff_sort,
      [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; i++) {
          pcg32 rng(seed, i);
          s[i] = gen_random_elem<T>(rng);
        }
      });
}

template <typename T>
bool check_sorted(span<T> s) {
  if (s.size() < cutoff_sort) {
    return std::is_sorted(s.begin(), s.end());
  }
  bool r1, r2;
  auto [s1, s2] = divide_two(s);
  mtbb::task_group tg;
  tg.run([&, &s1 = s1]{ r1 = check_sorted(s1); });
  tg.run([&, &s2 = s2]{ r2 = check_sorted(s2); });
  tg.wait();
  return r1 && r2 && (s1.back() <= s2.front());
}

void run() {
  std::vector<elem_t> a_vec(n_input);
  std::vector<elem_t> b_vec(n_input);

  span<elem_t> a(a_vec.data(), a_vec.size());
  span<elem_t> b(b_vec.data(), b_vec.size());

  for (int r = 0; r < n_repeats; r++) {
    fill_array(a);

    auto t0 = clock_gettime_ns();

    cilksort(a, b);

    auto t1 = clock_gettime_ns();

    printf("[%d] %ld ns", r, t1 - t0);

    if (verify_result) {
      printf(check_sorted(a) ? " - Result verified" : " - Wrong result");
    }

    printf("\n");
    fflush(stdout);
  }
}

void show_help_and_exit(int argc [[maybe_unused]], char** argv) {
  printf("Usage: %s [options]\n"
         "  options:\n"
         "    -n : Input size (size_t)\n"
         "    -r : # of repeats (int)\n"
         "    -s : cutoff count for serial sort (size_t)\n"
         "    -m : cutoff count for serial merge (size_t)\n"
         "    -v : verify the result (int)\n", argv[0]);
  exit(1);
}

int main(int argc, char** argv) {
  myth_init();

  int opt;
  while ((opt = getopt(argc, argv, "n:r:s:m:v:h")) != EOF) {
    switch (opt) {
      case 'n':
        n_input = atoll(optarg);
        break;
      case 'r':
        n_repeats = atoi(optarg);
        break;
      case 's':
        cutoff_sort = atoll(optarg);
        break;
      case 'm':
        cutoff_merge = atoll(optarg);
        break;
      case 'v':
        verify_result = atoi(optarg);
        break;
      case 'h':
      default:
        show_help_and_exit(argc, argv);
    }
  }

  setlocale(LC_NUMERIC, "en_US.UTF-8");
  printf("=============================================================\n"
         "[Cilksort (myth)]\n"
         "# of threads:                 %d\n"
         "Element:                      %s (%ld bytes)\n"
         "N:                            %ld\n"
         "# of repeats:                 %d\n"
         "Cutoff (cilksort):            %ld\n"
         "Cutoff (cilkmerge):           %ld\n"
         "Verify result:                %d\n"
         "=============================================================\n",
         myth_get_num_workers(),
         typename_str<elem_t>(), sizeof(elem_t), n_input, n_repeats,
         cutoff_sort, cutoff_merge, verify_result);
  printf("PID of the main worker: %d\n", getpid());
  printf("\n");
  fflush(stdout);

  run();

  myth_fini();
  return 0;
}
