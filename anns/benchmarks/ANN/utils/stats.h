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

#include <algorithm>
#if 0
#include "common/geometry.h"
#endif
#include "indexTools.h"
#include <set>

template<typename T>
auto graph_stats(ityr::global_span<Tvec_point<T>> v){
  struct stats {
    double      avg_deg;
    std::size_t max_deg;
  };
  return ityr::root_exec([=] {
    std::size_t sum = ityr::transform_reduce(
        ityr::execution::parallel_policy(1024),
        v.begin(), v.end(), ityr::reducer::plus<std::size_t>{},
        [](const Tvec_point<T>& p) { return size_of(p.out_nbh); });
    std::size_t maxDegree = ityr::transform_reduce(
        ityr::execution::parallel_policy(1024),
        v.begin(), v.end(), ityr::reducer::max<std::size_t>{},
        [](const Tvec_point<T>& p) { return size_of(p.out_nbh); });
    double avg_deg = sum/((double) v.size());
    return stats{avg_deg, maxDegree};
  });
}

template<typename T>
auto query_stats(ityr::global_span<Tvec_qpoint<T>> q){
  auto [vs1, vs2] = visited_stats(q);
  auto [ds1, ds2] = distance_stats(q);
  return std::make_tuple(ds1, ds2, vs1, vs2);
}

#if 0
template<typename T>
auto range_query_stats(parlay::sequence<Tvec_point<T>*> &q){
	auto pred = [&] (Tvec_point<T>* p) {return (p->ngh.size()==0);};
	auto pred1 = [&] (Tvec_point<T>* p) {return !pred(p);};
	auto zero_queries = parlay::filter(q, pred);
	auto nonzero_queries = parlay::filter(q, pred1);
	parlay::sequence<int> vn = visited_stats(nonzero_queries);
	parlay::sequence<int> dn = distance_stats(nonzero_queries);
	parlay::sequence<int> rn = rounds_stats(nonzero_queries);
	parlay::sequence<int> vz = visited_stats(zero_queries);
	parlay::sequence<int> dz = distance_stats(zero_queries);
	parlay::sequence<int> rz = rounds_stats(zero_queries);
	auto result = {rn, dn, vn, rz, dz, vz};
	return parlay::flatten(result);
}
#endif

template<typename T> 
auto visited_stats(ityr::global_span<Tvec_qpoint<T>> q){
  struct stats {
    size_t avg_visited;
    size_t tail_visited;
  };
  return ityr::root_exec([=] {
    ityr::global_vector_options global_vec_coll_opts(true, 1024);
    ityr::global_vector<std::size_t> visited(global_vec_coll_opts, q.size());
    ityr::transform(
        ityr::execution::parallel_policy(1024),
        q.begin(), q.end(), visited.begin(),
        [](const Tvec_qpoint<T>& p) { return p.visited; });
    ityr::sort(
        ityr::execution::parallel_policy(1024),
        visited.begin(), visited.end());
    size_t avg_visited = (int) ityr::reduce(
        ityr::execution::parallel_policy(1024),
        visited.begin(), visited.end())/((double) q.size());
    size_t tail_index = .99*((float) q.size());
    size_t tail_visited = visited[tail_index].get();
    return stats{avg_visited, tail_visited};
  });
}

template<typename T> 
auto distance_stats(ityr::global_span<Tvec_qpoint<T>> q){
  struct stats {
    size_t avg_dist;
    size_t tail_dist;
  };
  return ityr::root_exec([=] {
    ityr::global_vector_options global_vec_coll_opts(true, 1024);
    ityr::global_vector<std::size_t> dist(global_vec_coll_opts, q.size());
    ityr::transform(
        ityr::execution::parallel_policy(1024),
        q.begin(), q.end(), dist.begin(),
        [](const Tvec_qpoint<T>& p) { return p.dist_calls; });
    ityr::sort(
        ityr::execution::parallel_policy(1024),
        dist.begin(), dist.end());
    size_t avg_dist = (int) ityr::reduce(
        ityr::execution::parallel_policy(1024),
        dist.begin(), dist.end())/((double) q.size());
    size_t tail_index = .99*((float) q.size());
    size_t tail_dist = dist[tail_index].get();
    return stats{avg_dist, tail_dist};
  });
}

#if 0
template<typename T> 
parlay::sequence<size_t> rounds_stats(parlay::sequence<Tvec_point<T>*> &q){
	auto exp_stats = parlay::tabulate(q.size(), [&] (size_t i) {return q[i]->rounds;});
	parlay::sort_inplace(exp_stats);
	size_t avg_exps = (size_t) parlay::reduce(exp_stats)/((double) q.size());
	size_t tail_index = .99*((float) q.size());
	size_t tail_exps = exp_stats[tail_index];
	auto result = {avg_exps, tail_exps, exp_stats[exp_stats.size()-1]};
	return result;
}

void range_gt_stats(parlay::sequence<ivec_point> groundTruth){
  auto sizes = parlay::tabulate(groundTruth.size(), [&] (size_t i) {return groundTruth[i].coordinates.size();});
  parlay::sort_inplace(sizes);
  size_t first_nonzero_index;
  for(size_t i=0; i<sizes.size(); i++){ if(sizes[i] != 0){first_nonzero_index = i; break;}}
  auto nonzero_sizes = (sizes).cut(first_nonzero_index, sizes.size());
  auto sizes_sum = parlay::reduce(nonzero_sizes);
  float avg = static_cast<float>(sizes_sum)/static_cast<float>(nonzero_sizes.size());
  std::cout << "Among nonzero entries, the average number of matches is " << avg << std::endl;
  std::cout << "25th percentile: " << nonzero_sizes[.25*nonzero_sizes.size()] << std::endl;
  std::cout << "75th percentile: " << nonzero_sizes[.75*nonzero_sizes.size()] << std::endl;
  std::cout << "99th percentile: " << nonzero_sizes[.99*nonzero_sizes.size()] << std::endl;
  std::cout << "Max: " << nonzero_sizes[nonzero_sizes.size()-1] << std::endl;
}
#endif

