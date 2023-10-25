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
#include <cmath>
#if 0
#include "common/geometry.h"
#endif
#include "../utils/NSGDist.h"  
#include "../utils/types.h"
#include "../utils/beamSearch.h"
#include "../utils/indexTools.h"
#include "../utils/stats.h"
#include "../utils/parse_results.h"
#include "../utils/check_nn_recall.h"
#include "hcnng_index.h"

#ifndef ANNS_DATA_TYPE
#define ANNS_DATA_TYPE uint8_t
/* #define ANNS_DATA_TYPE int8_t */
/* #define ANNS_DATA_TYPE float */
#endif

extern bool report_stats;
template<typename T>
void ANN(ityr::global_vector<Tvec_point<T>> &v, int k, int mstDeg,
	 int num_clusters, int beamSizeQ, double cluster_size, double dummy,
	 ityr::global_vector<Tvec_qpoint<T>> &q, ityr::global_vector<ivec_point>& groundTruth, char* res_file, bool graph_built, bool mips) {

  timer t;
  using findex = hcnng_index<T>;
  unsigned d = (v[0].get().coordinates).size();
  double idx_time;
  if(!graph_built){
    auto v_copy = v;
    findex I(mstDeg, d, mips);
#if 0
     parlay::sequence<int> inserts = parlay::tabulate(v.size(), [&] (size_t i){
					    return static_cast<int>(i);});
#endif
    I.build_index(v_copy, num_clusters, cluster_size);
    idx_time = t.tick_s();
  } else{idx_time=0;}
  std::string name = "HCNNG";
  std::string params = "Trees = " + std::to_string(num_clusters);
  auto [avg_deg, max_deg] = graph_stats<T>(v);
  Graph G(name, params, v.size(), avg_deg, max_deg, idx_time);
  if (ityr::is_master()) {
    G.print();
  }
  search_and_parse<T>(G, v, q, groundTruth, res_file, mips);

}


#if 0
template<typename T>
void ANN(parlay::sequence<Tvec_point<T>*> v, int MSTdeg, int num_clusters, double cluster_size, double dummy2, bool graph_built, bool mips) {
  parlay::internal::timer t("ANN",report_stats); 
  { 
    unsigned d = (v[0]->coordinates).size();
    using findex = hcnng_index<T>;
    findex I(MSTdeg, d, mips);
    if(!graph_built){
      I.build_index(v, num_clusters, cluster_size);
      t.next("Built index");
    }
    if(report_stats){
      graph_stats(v);
      t.next("stats");
    }
  };
}
#endif
