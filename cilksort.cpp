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

#include "common.hpp"

enum class exec_t {
  Default = 0,
  StdSort = 1,
};

std::ostream& operator<<(std::ostream& o, const exec_t& e) {
  switch (e) {
    case exec_t::StdSort: o << "std_sort"; break;
    case exec_t::Default: o << "default"; break;
  }
  return o;
}

template <typename T>
std::string to_str(T x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

struct prof_event_user_sort_kernel : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_sort_kernel"; }
};

struct prof_event_user_merge_kernel : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_merge_kernel"; }
};

struct prof_event_user_sort : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_sort"; }
};

struct prof_event_user_merge : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_merge"; }
};

struct prof_event_user_binary_search : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_binary_search"; }
};

struct prof_event_user_copy : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_copy"; }
};

#ifndef ITYRBENCH_ELEM_TYPE
#define ITYRBENCH_ELEM_TYPE int
#endif
using elem_t = ITYRBENCH_ELEM_TYPE;

std::size_t n_input       = std::size_t(1) * 1024 * 1024;
int         n_repeats     = 10;
exec_t      exec_type     = exec_t::Default;
std::size_t cutoff_sort   = std::size_t(4) * 1024;
std::size_t cutoff_merge  = std::size_t(4) * 1024;
bool        verify_result = true;

template <typename T>
auto divide(const ityr::global_span<T>& s, typename ityr::global_span<T>::size_type at) {
  return std::make_pair(s.subspan(0, at), s.subspan(at, s.size() - at));
}

template <typename T>
auto divide_two(const ityr::global_span<T>& s) {
  return divide(s, s.size() / 2);
}

template <typename T>
std::size_t binary_search(ityr::global_span<T> s, const T& v) {
  // get() is internally called through ityr::global_ref
  auto it = std::lower_bound(s.begin(), s.end(), v);
  return it - s.begin();
}

template <typename T>
void cilkmerge(ityr::global_span<T> s1,
               ityr::global_span<T> s2,
               ityr::global_span<T> dest) {
  assert(s1.size() + s2.size() == dest.size());

  if (s1.size() < s2.size()) {
    // s2 is always smaller
    std::swap(s1, s2);
  }

  if (s2.size() == 0) {
    ITYR_PROFILER_RECORD(prof_event_user_copy);
    auto [s1_, dest_] =
      ityr::make_checkouts(s1  , ityr::checkout_mode::read,
                           dest, ityr::checkout_mode::write);
    std::copy(s1_.begin(), s1_.end(), dest_.begin());
    return;
  }

  if (dest.size() < cutoff_merge) {
    ITYR_PROFILER_RECORD(prof_event_user_merge);
    auto [s1_, s2_, dest_] =
      ityr::make_checkouts(s1  , ityr::checkout_mode::read,
                           s2  , ityr::checkout_mode::read,
                           dest, ityr::checkout_mode::write);
    {
      ITYR_PROFILER_RECORD(prof_event_user_merge_kernel);
      std::merge(s1_.begin(), s1_.end(), s2_.begin(), s2_.end(), dest_.begin());
    }
    return;
  }

  std::size_t split1 = (s1.size() + 1) / 2;
  std::size_t split2 = [&]() {
    ITYR_PROFILER_RECORD(prof_event_user_binary_search);
    return binary_search(s2, s1[split1 - 1].get());
  }();

  auto [s11  , s12  ] = divide(s1, split1);
  auto [s21  , s22  ] = divide(s2, split2);
  auto [dest1, dest2] = divide(dest, split1 + split2);

  ityr::parallel_invoke(
      cilkmerge<T>, std::make_tuple(s11, s21, dest1),
      cilkmerge<T>, std::make_tuple(s12, s22, dest2));
}

template <typename T>
void cilksort(ityr::global_span<T> a, ityr::global_span<T> b) {
  assert(a.size() == b.size());

  if (a.size() < cutoff_sort) {
    ITYR_PROFILER_RECORD(prof_event_user_sort);
    auto a_ = ityr::make_checkout(a, ityr::checkout_mode::read_write);
    {
      ITYR_PROFILER_RECORD(prof_event_user_sort_kernel);
      std::sort(a_.begin(), a_.end());
    }
    return;
  }

  auto [a12, a34] = divide_two(a);
  auto [b12, b34] = divide_two(b);

  auto [a1, a2] = divide_two(a12);
  auto [a3, a4] = divide_two(a34);
  auto [b1, b2] = divide_two(b12);
  auto [b3, b4] = divide_two(b34);

  ityr::parallel_invoke(
      cilksort<T>, std::make_tuple(a1, b1),
      cilksort<T>, std::make_tuple(a2, b2),
      cilksort<T>, std::make_tuple(a3, b3),
      cilksort<T>, std::make_tuple(a4, b4));

  ityr::parallel_invoke(
      cilkmerge<T>, std::make_tuple(a1, a2, b12),
      cilkmerge<T>, std::make_tuple(a3, a4, b34));

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
void fill_array(ityr::global_span<T> s) {
  static int counter = 0;
  auto seed = counter++;

  ityr::transform(
      ityr::execution::parallel_policy(cutoff_sort),
      ityr::count_iterator<std::size_t>(0),
      ityr::count_iterator<std::size_t>(s.size()),
      s.begin(),
      [=](std::size_t i) {
        pcg32 rng(seed, i);
        return gen_random_elem<T>(rng);
      });
}

template <typename T>
bool check_sorted(ityr::global_span<T> s) {
  if (s.size() <= 1) {
    return true;
  }
  // check s[i] <= s[i+1] for all i
  return ityr::is_sorted(
      ityr::execution::parallel_policy(cutoff_sort),
      s.begin(), s.end());
}

void show_help_and_exit(int argc [[maybe_unused]], char** argv) {
  if (ityr::is_master()) {
    printf("Usage: %s [options]\n"
           "  options:\n"
           "    -n : Input size (size_t)\n"
           "    -r : # of repeats (int)\n"
           "    -e : execution type (0: default, 1: std::sort())\n"
           "    -s : cutoff count for serial sort (size_t)\n"
           "    -m : cutoff count for serial merge (size_t)\n"
           "    -v : verify the result (int)\n", argv[0]);
  }
  exit(1);
}

int main(int argc, char** argv) {
  ityr::init();

  ityr::common::profiler::event_initializer<prof_event_user_sort_kernel>   ITYR_ANON_VAR;
  ityr::common::profiler::event_initializer<prof_event_user_merge_kernel>  ITYR_ANON_VAR;
  ityr::common::profiler::event_initializer<prof_event_user_sort>          ITYR_ANON_VAR;
  ityr::common::profiler::event_initializer<prof_event_user_merge>         ITYR_ANON_VAR;
  ityr::common::profiler::event_initializer<prof_event_user_binary_search> ITYR_ANON_VAR;
  ityr::common::profiler::event_initializer<prof_event_user_copy>          ITYR_ANON_VAR;

  set_signal_handlers();

  int opt;
  while ((opt = getopt(argc, argv, "n:r:e:s:m:v:h")) != EOF) {
    switch (opt) {
      case 'n':
        n_input = atoll(optarg);
        break;
      case 'r':
        n_repeats = atoi(optarg);
        break;
      case 'e':
        exec_type = exec_t(atoi(optarg));
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

  if (ityr::is_master()) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[Cilksort]\n"
           "# of processes:               %d\n"
           "Element:                      %s (%ld bytes)\n"
           "N:                            %ld\n"
           "# of repeats:                 %d\n"
           "Execution type:               %s\n"
           "Cutoff (cilksort):            %ld\n"
           "Cutoff (cilkmerge):           %ld\n"
           "Verify result:                %d\n"
           "-------------------------------------------------------------\n",
           ityr::n_ranks(), typename_str<elem_t>(), sizeof(elem_t), n_input, n_repeats,
           to_str(exec_type).c_str(), cutoff_sort, cutoff_merge, verify_result);

    printf("[Compile Options]\n");
    ityr::print_compile_options();
    printf("-------------------------------------------------------------\n");
    printf("[Runtime Options]\n");
    ityr::print_runtime_options();
    printf("=============================================================\n");
    printf("PID of the main worker: %d\n", getpid());
    printf("\n");
    fflush(stdout);
  }

  ityr::ori::global_ptr<elem_t> a_ptr = ityr::ori::malloc_coll<elem_t>(n_input);
  ityr::ori::global_ptr<elem_t> b_ptr = ityr::ori::malloc_coll<elem_t>(n_input);

  ityr::global_span<elem_t> a(a_ptr, n_input);
  ityr::global_span<elem_t> b(b_ptr, n_input);

  /* ityr::ito::adws_enable_steal_option::set(false); */

  for (int r = 0; r < n_repeats; r++) {
    ityr::root_exec([=] {
      fill_array(a);
    });

    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    if (exec_type == exec_t::StdSort) {
      ityr::root_exec([=] {
        std::sort(a.begin(), a.end());
      });
    } else {
      ityr::root_exec([=] {
        cilksort(a, b);
      });
    }

    auto t1 = ityr::gettime_ns();

    ityr::profiler_end();

    if (ityr::is_master()) {
      printf("[%d] %ld ns", r, t1 - t0);
    }

    if (verify_result) {
      bool success = ityr::root_exec([=] {
        return check_sorted(a);
      });
      if (ityr::is_master()) {
        printf(success ? " - Result verified" : " - Wrong result");
      }
    }

    if (ityr::is_master()) {
      printf("\n");
      fflush(stdout);
    }

    ityr::profiler_flush();
  }

  ityr::ori::free_coll<elem_t>(a_ptr);
  ityr::ori::free_coll<elem_t>(b_ptr);

  ityr::fini();
  return 0;
}
