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
#include "types.h"
#include "parse_results.h"
#include "beamSearch.h"
#include "csvfile.h"

template<typename T>
nn_result checkRecall(
        ityr::global_span<Tvec_point<T>> v,
        ityr::global_span<Tvec_qpoint<T>> q,
        ityr::global_span<ivec_point> groundTruth,
        int k,
        int beamQ,
        float cut,
        unsigned d,
        bool random,
        int limit,
        int start_point,
        bool mips) {
  timer t;
  int r = 10;
  float query_time;
  if(random){
    beamSearchRandom(q, v, beamQ, k, d, mips, cut, limit);
    t.tick_s();
    beamSearchRandom(q, v, beamQ, k, d, mips, cut, limit);
    query_time = t.tick_s();
  }else{
#if 0
    searchAll(q, v, beamQ, k, d, v[start_point], mips, cut, limit);
    t.next_time();
    searchAll(q, v, beamQ, k, d, v[start_point], mips, cut, limit);
    query_time = t.next_time();
#else
    (void)start_point;
    std::cout << "Error: only random search is supported" << std::endl;
    abort();
#endif
  }
  float recall = 0.0;
  bool dists_present = (groundTruth[0].get().distances.size() != 0);
  if (groundTruth.size() > 0 && !dists_present) {
#if 0
    size_t n = q.size();
    int numCorrect = 0;
    for(int i=0; i<n; i++){
      std::set<int> reported_nbhs;
      for(int l=0; l<r; l++) reported_nbhs.insert((q[i]->ngh)[l]);
      for(int l=0; l<r; l++){
	      if (reported_nbhs.find((groundTruth[i].coordinates)[l]) != reported_nbhs.end()){
          numCorrect += 1;
      }
      }
    }
    recall = static_cast<float>(numCorrect)/static_cast<float>(r*n);
#else
    std::cout << "Error: distances should be available" << std::endl;
    abort();
#endif
  }else if(groundTruth.size() > 0 && dists_present){
    recall = ityr::root_exec([=] {
      size_t n = q.size();
      int numCorrectAll = ityr::transform_reduce(
          ityr::execution::parallel_policy(1024),
          q.begin(), q.end(), groundTruth.begin(),
          ityr::reducer::plus<int>{},
          [=](const Tvec_qpoint<T>& qp, const ivec_point& gt) {
            std::vector<int> results_with_ties;
            for(int l=0; l<r; l++) results_with_ties.push_back(gt.coordinates[l]);
            float last_dist = gt.distances[r-1];
            for(std::size_t l=r; l<gt.coordinates.size(); l++){
              if(gt.distances[l] == last_dist){ 
                results_with_ties.push_back(gt.coordinates[l]);
              }
            }
            std::set<int> reported_nbhs;
            auto ngh_cs = ityr::make_checkout(qp.ngh.data(), qp.ngh.size(), ityr::checkout_mode::read);
            for(int l=0; l<r; l++) reported_nbhs.insert(ngh_cs[l]);

            int numCorrect = 0;
            for(std::size_t l=0; l<results_with_ties.size(); l++){
                    if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end()){
                numCorrect += 1;
              }
            }
            return numCorrect;
          });
      return static_cast<float>(numCorrectAll)/static_cast<float>(r*n);
    });
  }
  float QPS = q.size()/query_time;
  auto stats = query_stats(q);
  nn_result N(recall, stats, QPS, k, beamQ, cut, q.size());
  return N;
}

#if 0
void write_to_csv(std::string csv_filename, parlay::sequence<float> buckets, 
  parlay::sequence<nn_result> results, Graph G){
  csvfile csv(csv_filename);
  csv << "GRAPH" << "Parameters" << "Size" << "Build time" << "Avg degree" << "Max degree" << endrow;
  csv << G.name << G.params << G.size << G.time << G.avg_deg << G.max_deg << endrow;
  csv << endrow;
  csv << "Num queries" << "Target recall" << "Actual recall" << "QPS" << "Average Cmps" << 
    "Tail Cmps" << "Average Visited" << "Tail Visited" << "k" << "Q" << "cut" << endrow;
  for(int i=0; i<results.size(); i++){
    nn_result N = results[i];
    csv << N.num_queries << buckets[i] << N.recall << N.QPS << N.avg_cmps << N.tail_cmps <<  
      N.avg_visited << N.tail_visited << N.k << N.beamQ << N.cut << endrow;
  }
  csv << endrow;
  csv << endrow;
}
#endif

inline std::vector<int> calculate_limits(size_t avg_visited){
  std::vector<int> L(9);
  for(float i=1; i<10; i++){
    L[i-1] = (int) (i *((float) avg_visited) * .1);
  }
  /* auto limits = parlay::remove_duplicates(L); */
  /* return limits; */
  std::stable_sort(L.begin(), L.end());
  L.erase(std::unique(L.begin(), L.end()), L.end());
  return L;
}    

template<typename T>
void search_and_parse(Graph G, ityr::global_span<Tvec_point<T>> v, ityr::global_span<Tvec_qpoint<T>> q, 
    ityr::global_span<ivec_point> groundTruth, char* res_file, bool mips, bool fast_check, bool random=true, int start_point=0){
  unsigned d = v[0].get().coordinates.size();

  if (fast_check) {
    nn_result result1 = checkRecall(v, q, groundTruth, 100, 500, 1.25, d, random, -1, start_point, mips);
    if (ityr::is_master()) {
      result1.print();
    }

    int limit = (int) ((float) result1.avg_visited * 0.05);
    nn_result result2 = checkRecall(v, q, groundTruth, 10, 15, 1.14, d, random, limit, start_point, mips);
    if (ityr::is_master()) {
      result2.print();
    }

  } else {
    std::vector<nn_result> results;
    std::vector<int> beams = {15, 20, 30, 50, 75, 100, 125, 250, 500};
    std::vector<int> allk = {10, 15, 20, 30, 50, 100};
    std::vector<float> cuts = {1.1, 1.125, 1.15, 1.175, 1.2, 1.25};
    for (float cut : cuts)
      for (float Q : beams) 
        results.push_back(checkRecall(v, q, groundTruth, 10, Q, cut, d, random, -1, start_point, mips));

    for (float cut : cuts)
      for (int kk : allk)
        results.push_back(checkRecall(v, q, groundTruth, kk, 500, cut, d, random, -1, start_point, mips));

    // check "limited accuracy"
    std::vector<int> limits = calculate_limits(results[0].avg_visited);
    for(int l : limits){
      results.push_back(checkRecall(v, q, groundTruth, 10, 15, 1.14, d, random, l, start_point, mips));
    }

    // check "best accuracy"
    results.push_back(checkRecall(v, q, groundTruth, 100, 1000, 10.0, d, random, -1, start_point, mips));

    if (ityr::is_master()) {
      std::vector<float> buckets = {.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .73, .75, .77, .8, .83, .85, .87, .9, .93, .95, .97, .99, .995, .999};
      auto [res, ret_buckets] = parse_result(results, buckets);
      if(res_file != NULL) {
#if 0
        write_to_csv(std::string(res_file), ret_buckets, res, G);
#else
        (void)G;
        std::cout << "Error: CSV output is not supported" << std::endl;
        abort();
#endif
      }
    }
  }
}

