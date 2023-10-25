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
#include "../utils/clusterEdge.h"
#include <random>
#include <set>
#include <math.h>
#include <queue>

extern bool report_stats;

template<typename T>
struct hcnng_index{
	int maxDeg;
	unsigned d;
	bool mips;
	using tvec_point = Tvec_point<T>;
#if 0
	using slice_tvec = decltype(make_slice(parlay::sequence<tvec_point*>()));
#endif
	using edge = std::pair<int, int>;
	using labelled_edge = std::pair<edge, float>;
	using pid = std::pair<int, float>;

	hcnng_index(int md, unsigned dim, bool m) : maxDeg(md), d(dim), mips(m) {}

	float Distance(T* p, T* q, unsigned d){
		if(mips) return mips_distance(p, q, d);
		else return distance(p, q, d);
	}

	void remove_all_duplicates(ityr::global_span<tvec_point> v){
          ityr::root_exec([=] {
            ityr::for_each(
              ityr::execution::parallel_policy(1024),
              ityr::make_global_iterator(v.begin(), ityr::checkout_mode::read),
              ityr::make_global_iterator(v.end(),   ityr::checkout_mode::read),
              [=](tvec_point& p) {
                cluster<T>::remove_edge_duplicates(p);
              });
          });
	}


	void build_index(ityr::global_span<tvec_point> v, int cluster_rounds, size_t cluster_size){ 
          if (ityr::is_master()) {
		std::cout << "Mips: " << mips << std::endl;
          }
		clear(v); 
		cluster<T> C(d, mips);
		C.multiple_clustertrees(v, cluster_size, cluster_rounds, d, maxDeg);
		remove_all_duplicates(v);
	}
	
};
