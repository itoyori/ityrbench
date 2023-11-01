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
#include <random>
#include <set>
#include <math.h>
#include <functional>
#include <queue>

struct prof_event_user_mst_pre : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_MST_pre"; }
};

struct prof_event_user_mst_post : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_MST_post"; }
};

#if 0
std::pair<size_t, size_t> select_two_random(parlay::sequence<size_t>& active_indices,
	parlay::random& rnd) {
	size_t first_index = rnd.ith_rand(0) % active_indices.size(); 
	size_t second_index_unshifted = rnd.ith_rand(1) % (active_indices.size()-1);
	size_t second_index = (second_index_unshifted < first_index) ?
	second_index_unshifted : (second_index_unshifted + 1);

	return {active_indices[first_index], active_indices[second_index]};
}
#endif

struct DisjointSet{
        std::vector<int> parent;
	std::vector<int> rank;
	size_t N; 

	DisjointSet(size_t size) : parent(size), rank(size), N(size) {
          // TODO: parallelize?
          for (std::size_t i = 0; i < N; i++) {
            parent[i] = i;
            rank[i] = i;
          }
	}

	void _union(int x, int y){
		int xroot = parent[x];
		int yroot = parent[y];
		int xrank = rank[x];
		int yrank = rank[y];
		if(xroot == yroot)
			return;
		else if(xrank < yrank)
			parent[xroot] = yroot;
		else{
			parent[yroot] = xroot;
			if(xrank == yrank)
				rank[xroot] = rank[xroot] + 1;
		}
	}

	int find(int x){
		if(parent[x] != x)
			parent[x] = find(parent[x]);
		return parent[x];
	}

	void flatten(){
		for(std::size_t i=0; i<N; i++) find(i);
	}

	bool is_full(){
		flatten();
#if 0
		parlay::sequence<bool> truthvals(N);
		parlay::parallel_for(0, N, [&] (size_t i){
			truthvals[i] = (parent[i]==parent[0]);
		});
		auto ff = [&] (bool a) {return not a;};
		auto filtered = parlay::filter(truthvals, ff);
		if(filtered.size()==0) return true;
		return false;
#else
          // TODO: parallelize?
          for (std::size_t i = 0; i < N; i++) {
            if (parent[i] != parent[0]) {
              return false;
            }
          }
          return true;
#endif
	}

};

template<typename T>
struct cluster{
	unsigned d; 
	bool mips;
	using tvec_point = Tvec_point<T>;
	using edge = std::pair<int, int>;
	using labelled_edge = std::pair<edge, float>;

	cluster(unsigned dim, bool m): d(dim), mips(m) {}

	float Distance(T* p, T* q, unsigned d) const {
		if(mips) return mips_distance(p, q, d);
		else return distance(p, q, d);
	}

	//inserts each edge after checking for duplicates
	void process_edges(ityr::global_span<tvec_point> v, std::vector<edge> edges) const {
#if 0
		auto grouped = parlay::group_by_key(edges);
		for(auto pair : grouped){
			auto [index, candidates] = pair;
			for(auto c : candidates){
				if(size_of(v[index]->out_nbh) < maxDeg){
					add_nbh(c, v[index]);
				}else{
					remove_edge_duplicates(v[index]);
					add_nbh(c, v[index]);
				}
			}
		}
#endif
                // TODO: parallelize?
                std::stable_sort(edges.begin(), edges.end());
                auto v_cs = ityr::make_checkout(v.data(), v.size(), ityr::checkout_mode::read);
		int maxDeg = v_cs[0].out_nbh.size();
                for (auto [i, c] : edges) {
                  if(size_of(v_cs[i].out_nbh) < maxDeg){
                    add_nbh(c, v_cs[i]);
                  }else{
                    // TODO: why does this never overflow?
                    remove_edge_duplicates(v_cs[i]);
                    add_nbh(c, v_cs[i]);
                  }
                }
	}

	static void remove_edge_duplicates(tvec_point& p){
          std::vector<int> points;
          {
            auto out_nbh_cs = ityr::make_checkout(p.out_nbh, ityr::checkout_mode::read);
		for(std::size_t i=0; i<out_nbh_cs.size(); i++){
                  if (out_nbh_cs[i] == -1) break;
                  points.push_back(out_nbh_cs[i]);
		}
          }
		/* auto np = parlay::remove_duplicates(points); */
                // TODO: parallelize?
                std::stable_sort(points.begin(), points.end());
                points.erase(std::unique(points.begin(), points.end()), points.end());
		add_out_nbh(points, p);
	}

	int generate_index(int N, int i) const {
		return (N*(N-1) - (N-i)*(N-i-1))/2;
	}
	
	//parameters dim and K are just to interface with the cluster tree code
	void MSTk(ityr::global_span<tvec_point> v,
		unsigned dim, int K) const {
		//preprocessing for Kruskal's
		int N = v.size();
		size_t m = 10;
		auto less = [&] (labelled_edge a, labelled_edge b) {return a.second < b.second;};

                ityr::global_vector<labelled_edge> flat_edges = ityr::transform_reduce(
                    ityr::execution::par,
                    ityr::count_iterator<int>(0),
                    ityr::count_iterator<int>(N),
                    ityr::reducer::vec_concat<labelled_edge>{},
                    [=](int i) {
                      ITYR_PROFILER_RECORD(prof_event_user_mst_pre);

                      std::priority_queue<labelled_edge, std::vector<labelled_edge>, decltype(less)> Q(less);
                      auto v_cs = ityr::make_checkout(v.data(), v.size(), ityr::checkout_mode::read);
                      for(int j=0; j<N; j++){
                              if(j!=i){
                                      float dist_ij = Distance(v_cs[i].coordinates.begin(), v_cs[j].coordinates.begin(), dim);
                                      if(Q.size() >= m){
                                              float topdist = Q.top().second;
                                              if(dist_ij < topdist){
                                                      labelled_edge e;
                                                      if(i<j) e = std::make_pair(std::make_pair(i,j), dist_ij);
                                                      else e = std::make_pair(std::make_pair(j, i), dist_ij);
                                                      Q.pop();
                                                      Q.push(e);
                                              }
                                      }else{
                                              labelled_edge e;
                                              if(i<j) e = std::make_pair(std::make_pair(i,j), dist_ij);
                                              else e = std::make_pair(std::make_pair(j, i), dist_ij);
                                              Q.push(e);
                                      }
                              }
                      }
                      ityr::global_vector<labelled_edge> edges(m);
                      for(std::size_t j=0; j<m; j++){edges[j] = Q.top(); Q.pop();}
                      return edges;
                    });

                ITYR_PROFILER_RECORD(prof_event_user_mst_post);

		// std::cout << flat_edges.size() << std::endl;
		auto less_dup = [&] (labelled_edge a, labelled_edge b){
			auto dist_a = a.second;
			auto dist_b = b.second;
			if(dist_a == dist_b){
				int i_a = a.first.first;
				int j_a = a.first.second;
				int i_b = b.first.first;
				int j_b = b.first.second;
				if((i_a==i_b) && (j_a==j_b)){
					return true; // TODO: really? I think this should return false...
				} else{
					if(i_a != i_b) return i_a < i_b;
					else return j_a < j_b;
				}
			}else return (dist_a < dist_b);
		};
		/* auto labelled_edges = parlay::remove_duplicates_ordered(flat_edges, less_dup); */
                // TODO: parallelize?
                auto flat_edges_cs = ityr::make_checkout(flat_edges.data(), flat_edges.size(), ityr::checkout_mode::read_write);
                std::stable_sort(flat_edges_cs.begin(), flat_edges_cs.end(), less_dup);
                auto flat_edges_end = std::unique(flat_edges_cs.begin(), flat_edges_cs.end());
		// parlay::sort_inplace(labelled_edges, less);
                std::vector<int> degrees(N, 0);
                std::vector<edge> MST_edges;
		//modified Kruskal's algorithm
		DisjointSet *disjset = new DisjointSet(N);
                auto v_cs = ityr::make_checkout(v.data(), v.size(), ityr::checkout_mode::read);
		for(long i=0; i<flat_edges_end - flat_edges_cs.begin(); i++){
			labelled_edge e_l = flat_edges_cs[i];
			edge e = e_l.first;
			if((disjset->find(e.first) != disjset->find(e.second)) && (degrees[e.first]<K) && (degrees[e.second]<K)){
				MST_edges.push_back(std::make_pair(e.first, v_cs[e.second].id));
				MST_edges.push_back(std::make_pair(e.second, v_cs[e.first].id));
				degrees[e.first] += 1;
				degrees[e.second] += 1;
				disjset->_union(e.first, e.second);
			}
			if(i%N==0){
				if(disjset->is_full()){
					break;
				}
			}
		}
		delete disjset;
		process_edges(v, MST_edges);
	}

	bool tvec_equal(tvec_point a, tvec_point b, unsigned d) const {
		for(int i=0; i<d; i++){
			if(a.coordinates[i] != b.coordinates[i]){
				return false;
			}
		}
		return true;
	}

        template <typename Rng>
	void recurse(ityr::global_span<tvec_point> v,
		Rng& rng, size_t cluster_size, 
		unsigned dim, int K, const tvec_point& first, const tvec_point& second) const {

          auto v_mid = ityr::partition(
              ityr::execution::parallel_policy(1024),
              v.begin(), v.end(), [=, *this](const tvec_point& p) {
                float dist_first = distance(p.coordinates.begin(), first.coordinates.begin(), d);
                float dist_second = distance(p.coordinates.begin(), second.coordinates.begin(), d);
                return dist_first <= dist_second;
              });

          auto v_left = v.subspan(0, v_mid - v.begin());
          auto v_right = v.subspan(v_mid - v.begin(), v.end() - v_mid);

          auto left_rng = rng.split();
          auto right_rng = rng.split();

		if(v_left.size() == 1) {
			random_clustering(v, right_rng, cluster_size, dim, K);
		}
		else if(v_right.size() == 1){
			random_clustering(v, left_rng, cluster_size, dim, K);
		}
		else{
                  ityr::parallel_invoke(
				[=, *this]() mutable {random_clustering(v_left, left_rng, cluster_size, dim, K);}, 
				[=, *this]() mutable {random_clustering(v_right, right_rng, cluster_size, dim, K);}
			);
		}
	}

        template <typename Rng>
	void random_clustering(ityr::global_span<tvec_point> v,
		Rng& rng, size_t cluster_size, unsigned dim, int K) const {
		if(v.size() < cluster_size) MSTk(v, dim, K);
		else{
                        std::uniform_int_distribution<int> uni(0,v.size()-1);
                        int f = uni(rng);
                        int s = uni(rng);

    		tvec_point first = v[f].get();
    		tvec_point second = v[s].get();

			if(tvec_equal(first, second, dim)){
#if 0
				// std::cout << "Equal points selected, splitting evenly" << std::endl;
				parlay::sequence<size_t> closer_first;
				parlay::sequence<size_t> closer_second;
				for(int i=0; i<active_indices.size(); i++){
					if(i<active_indices.size()/2) closer_first.push_back(active_indices[i]);
					else closer_second.push_back(active_indices[i]);
				}
#endif
                                auto left_rng = rng.split();
                                auto right_rng = rng.split();

                                std::size_t mid = v.size() / 2;
				ityr::parallel_invoke(
					[=, *this]() mutable {random_clustering(v.subspan(0, mid), left_rng, cluster_size, dim, K);}, 
					[=, *this]() mutable {random_clustering(v.subspan(mid, v.size() - mid), right_rng, cluster_size, dim, K);}
				);
			} else{
				recurse(v, rng, cluster_size, dim, K, first, second);
			}
		}
	}

        template <typename Rng>
	void random_clustering_wrapper(ityr::global_span<tvec_point> v, size_t cluster_size, 
		unsigned dim, int K, Rng& rng) const {
#if 0
		std::random_device rd;    
  		std::mt19937 rng(rd());   
  		std::uniform_int_distribution<int> uni(0,v.size()); 
    	parlay::random rnd(uni(rng));
    	auto active_indices = parlay::tabulate(v.size(), [&] (size_t i) { return i; });
#endif
            random_clustering(v, rng, cluster_size, dim, K);
	}

	void multiple_clustertrees(ityr::global_span<tvec_point> v, size_t cluster_size, int num_clusters,
		unsigned dim, int K, int bound = 0) const {
          (void)bound;
          ityr::root_exec([=, *this] {
            uint64_t seed = 42;
            ityr::default_random_engine rng(seed);
		for(int i=0; i<num_clusters; i++){
			random_clustering_wrapper(v, cluster_size, dim, K, rng);
		}
          });
	}
};
