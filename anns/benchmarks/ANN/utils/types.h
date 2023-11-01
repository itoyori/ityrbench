// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TYPES
#define TYPES

#include <algorithm>
#include "../../../../common.hpp"

template <typename T>
using raw_span = ityr::common::span<T>;

//for a file in .fvecs or .bvecs format, but extendible to other types
template<typename T>
struct Tvec_point {
  using value_type = T;
  int id;
  raw_span<T> coordinates;
  ityr::global_span<int> out_nbh; 
  Tvec_point() = default;
  Tvec_point(const Tvec_point&) = default;
};

// query points
template<typename T>
struct Tvec_qpoint {
  using value_type = T;
  int id;
  size_t visited;
  size_t dist_calls;  
  int rounds;
  raw_span<T> coordinates;
  ityr::global_span<int> out_nbh; 
  ityr::global_span<int> new_nbh; 
  ityr::global_vector<int> ngh;
};



//for an ivec file, which contains the ground truth
//only info needed is the coordinates of the nearest neighbors of each point
struct ivec_point {
  using value_type = int;
  int id;
  raw_span<int> coordinates;
  raw_span<float> distances;
};


struct timer {
public:
  timer() {
    reset();
  }

  ityr::wallclock_t tick_ns() {
    auto t = ityr::gettime_ns();
    auto duration = t - prev_time_;
    prev_time_ = t;
    return duration;
  }

  double tick_s() {
    return tick_ns() / 1000000000.0;
  }

  ityr::wallclock_t total_duration_ns() const {
    return ityr::gettime_ns() - init_time_;
  }

  double total_duration_s() const {
    return total_duration_ns() / 1000000000.0;
  }

  void reset() {
    auto t = ityr::gettime_ns();
    prev_time_ = t;
    init_time_ = t;
  }

private:
  ityr::wallclock_t prev_time_;
  ityr::wallclock_t init_time_;
};

template <typename Iterator, typename Compare, typename T>
inline std::pair<Iterator, Iterator>
partition_three(Iterator first,
                Iterator last,
                Compare  comp,
                const T& pivot) {
  auto d = std::distance(first, last);

  if (d <= 1024 * 16) {
    auto [css, its] = ityr::internal::checkout_global_iterators(d, first);
    auto first_ = std::get<0>(its);

    auto l = first_;
    auto m = first_;
    auto r = std::next(first_, d);

    while (m < r) {
      if (comp(*m, pivot)) {
        std::swap(*l, *m);
        l++;
        m++;
      } else if (comp(pivot, *m)) {
        r--;
        std::swap(*m, *r);
      } else {
        m++;
      }
    }

    return std::make_pair(std::next(first, std::distance(first_, l)),
                          std::next(first, std::distance(first_, m)));
  }

  auto mid = std::next(first, d / 2);

  auto [mm1, mm2] = ityr::parallel_invoke(
      [=]() { return partition_three(first, mid , comp, pivot); },
      [=]() { return partition_three(mid  , last, comp, pivot); });

  auto [m11, m12] = mm1;
  auto [m21, m22] = mm2;

  auto me = ityr::rotate(
      ityr::execution::parallel_policy(1024 * 16),
      m11, mid, m22);

  return std::make_pair(m11 + (m21 - mid), me + (m12 - m11));
}

template <typename Iterator, typename Compare, typename GroupOp>
inline void groupby(Iterator first,
                    Iterator last,
                    Compare  comp,
                    GroupOp  group_op) {
  if (first == last) return;

  auto d = std::distance(first, last);

  if (d == 1) {
    group_op(first, last);
    return;
  }

  // FIXME: assumes global_ref
  auto pivot = (*first).get();

  auto mm = partition_three(first, last, comp, pivot);
  auto m1 = mm.first;
  auto m2 = mm.second;

  assert(m1 < m2);

  ityr::parallel_invoke(
      [=] { groupby(first, m1, comp, group_op); },
      [=] { group_op(m1, m2); },
      [=] { groupby(m2, last, comp, group_op); });
}

#endif
