#ifndef _HNSW_HPP
#define _HNSW_HPP

#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <atomic>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <set>
#include <iterator>
#include <type_traits>
#include <limits>
#include <thread>
#include "../../../../common.hpp"
// #include "parallelize.h"
#include "debug.hpp"
#define DEBUG_OUTPUT 0
#if DEBUG_OUTPUT
#define debug_output(...) my_printf(__VA_ARGS__)
#else
#define debug_output(...) do{[](...){}(__VA_ARGS__);}while(0)
#endif // DEBUG_OUTPUT

inline void my_printf(const char* format, ...) {
  if (ityr::is_spmd()) {
    if (!ityr::is_master()) return;
  } else if (ityr::is_root()) {
    ityr::migrate_to_master();
  }
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
  fflush(stdout);
}

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

namespace ANN{

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

enum class type_metric{
	L2, ANGULAR, DOT
};

struct point{
	float x, y;
};

template<typename U, template<typename> class Allocator=std::allocator>
class HNSW
{
	using T = typename U::type_point;
	typedef uint32_t node_id;
public:
	/*
		Construct from the vectors [begin, end).
		std::iterator_trait<Iter>::value_type ought to be convertible to T
		dim: 				vector dimension
		m_l: 				control the # of levels (larger m_l leads to more layer)
		m: 					max degree
		ef_construction:	beam size during the construction
		alpha:				parameter of the heuristic (similar to the one in vamana)
		batch_base: 		growth rate of the batch size (discarded because of two passes)
	*/
	template<typename Iter>
	HNSW(Iter begin, Iter end, uint32_t dim, float m_l=1, uint32_t m=100, uint32_t ef_construction=50, float alpha=5, float batch_base=2, bool do_fixing=false);

#if 0
	/*
		Construct from the saved model
		getter(i) returns the actual data (convertible to type T) of the vector with id i
	*/
	template<typename G>
	HNSW(const std::string &filename_model, G getter);
#endif

	ityr::global_vector<std::pair<uint32_t,float>> search(const T &q, uint32_t k, uint32_t ef, search_control ctrl={});
	// parlay::sequence<std::tuple<uint32_t,uint32_t,float>> search_ex(const T &q, uint32_t k, uint32_t ef, uint64_t verbose=0);
#if 0
	// save the current model to a file
	void save(const std::string &filename_model) const;
#endif
public:
	typedef uint32_t type_index;

	struct node{
		// uint32_t id;
		uint32_t level;
                ityr::global_vector<ityr::global_vector<node_id>> neighbors;
		T data; // point
                node() {}
                node(uint32_t level, ityr::global_vector<ityr::global_vector<node_id>>&& neighbors, const T& data)
                  : level(level), neighbors(std::move(neighbors)), data(data) {}
	};

	struct dist{
		float d;
		node_id u;
	};

	struct dist_ex : dist
	{
		uint32_t depth;
	};

	struct nearest{
		constexpr bool operator()(const dist &lhs, const dist &rhs) const{
			return lhs.d>rhs.d;
		}
	};

	struct farthest{
		constexpr bool operator()(const dist &lhs, const dist &rhs) const{
			return lhs.d<rhs.d;
		}
	};

/*
	struct cmp_id{
		constexpr bool operator()(const dist &lhs, const dist &rhs) const{
			return U::get_id(get_node(lhs.u).data)<U::get_id(get_node(rhs.u).data);
		}
	};
*/

        ityr::global_vector<node_id> entrance; // To init
	// auto m, max_m0, m_L; // To init
	uint32_t dim;
	float m_l;
	uint32_t m;
	// uint32_t level_max = 30; // To init
	uint32_t ef_construction;
	float alpha;
	uint32_t n;
	Allocator<node> allocator;
        ityr::global_vector<node> node_pool;

        mutable std::size_t total_visited = 0;
        mutable std::size_t total_eval = 0;
        mutable std::size_t total_size_C = 0;
        mutable std::size_t total_range_candidate = 0;

#if 0
	// `set_neighbourhood` will consume `vNewConn`
	static void set_neighbourhood(node &u, uint32_t level, parlay::sequence<node_id>& vNewConn)
	{
		u.neighbors[level] = std::move(vNewConn);
	}
#endif

/*
	static void add_connection(parlay::sequence<node_id> &neighbors, node &u, uint32_t level)
	{
		for(auto pv : neighbors)
		{
			assert(&u!=pv);
			pv->neighbors[level].push_back(&u);
			u.neighbors[level].push_back(pv);
		}
	}
*/
	// node* insert(const T &q, uint32_t id);
	template<typename Iter, typename Rng>
	void insert(Iter begin, Iter end, bool from_blank, Rng&& rng);

#if 0
	template<typename Queue>
	void select_neighbors_simple_impl(const T &u, Queue &C, uint32_t M)
	{
		/*
		list res;
		for(uint32_t i=0; i<M; ++i)
		{
			res.insert(C.pop_front());
		}
		return res;
		*/
		(void)u;
		parlay::sequence<typename Queue::value_type> tie;
		float dist_tie = 1e20;
		while(C.size()>M)
		{
			const auto &t = C.top();
			if(t.d+1e-6<dist_tie) // t.d<dist_tie
			{
				dist_tie = t.d;
				tie.clear();
			}
			if(fabs(dist_tie-t.d)<1e-6) // t.d==dist_tie
				tie.push_back(t);
			C.pop();
		}
		if(fabs(dist_tie-C.top().d)<1e-6) // C.top().d==dist_tie
			while(!tie.empty())
			{
			//	C.push({dist_tie,tie.back()});
				C.push(tie.back());
				tie.pop_back();
			}
	}

	template<typename Queue>
	auto select_neighbors_simple(const T &u, const Queue &C, uint32_t M)
	{
		// The parameter C is intended to be copy constructed
		/*
		select_neighbors_simple_impl(u, C, M);
		return C;
		*/
		// auto R = parlay::sort(C, farthest());
		auto R = C;
		
		if(R.size()>M)
		{
			std::nth_element(R.begin(), R.begin()+M, R.end(), farthest());
			R.resize(M);
		}
		
		std::sort(R.begin(), R.end(), farthest());
		// if(R.size()>M) R.resize(M);
		/*
		uint32_t size_R = std::min(C.size(),M);
		parlay::sequence<node*> R;
		R.reserve(size_R);
		for(const auto &e : C)
			R.push_back(e.u);
		*/

		return R;
	}
#endif

	// To optimize
	auto select_neighbors_heuristic(const T &u, 
		/*const std::priority_queue<dist,parlay::sequence<dist>,farthest> &C*/
		const std::vector<dist> &C, uint32_t M,
		uint32_t level, bool extendCandidate, bool keepPrunedConnections)
	{
		(void)extendCandidate;

		// std::priority_queue<dist,parlay::sequence<dist>,farthest> C_cp=C, W_d;
                std::vector<dist> W_d;
		std::set<node_id> W_tmp;
		// while(!C_cp.empty())
                for(auto &e : C) // TODO: add const?
                {
                        // auto &e = C_cp.top();
                        W_tmp.insert(e.u);
                        if(extendCandidate)
                        {
                          auto e_cs = ityr::make_checkout(&node_pool[e.u], 1, ityr::checkout_mode::read);

                          auto nbh_v_cs = ityr::make_checkout(&e_cs[0].neighbors[level], 1, ityr::checkout_mode::read);
                          auto &nbh_v = nbh_v_cs[0];

                                for(node_id e_adj : nbh_v)
                                {
                                        // if(e_adj==nullptr) continue; // TODO: check
                                        if(W_tmp.find(e_adj)==W_tmp.end())
                                                W_tmp.insert(e_adj);
                                }
                        }
                        // C_cp.pop();
                }

		// std::priority_queue<dist,parlay::sequence<dist>,nearest> W;
		std::vector<dist> W;
		W.reserve(W_tmp.size());
		for(node_id p : W_tmp) {
                  auto e_cs = ityr::make_checkout(&node_pool[p], 1, ityr::checkout_mode::read);
			W.push_back({U::distance(e_cs[0].data,u,dim), p});
                }
		std::sort(W.begin(), W.end(), farthest());
		/*
		for(auto &e : W_tmp)
			W.push(e);
		*/
		W_tmp.clear();

		std::vector<node_id> R;
		std::set<node_id> nbh;
		// while(W.size()>0 && R.size()<M)
		for(const auto &e : W)
		{
			if(R.size()>=M) break;
			// const auto e = W.top();
			// W.pop();
			const auto d_q = e.d;

                        auto e_cs = ityr::make_checkout(&node_pool[e.u], 1, ityr::checkout_mode::read);

			bool is_good = true;
			for(const auto &r : R)
			{
                          auto r_cs = ityr::make_checkout(&node_pool[r], 1, ityr::checkout_mode::read);
				const auto d_r = U::distance(e_cs[0].data, r_cs[0].data, dim);
				//if(d_r*(level+1)>d_q*alpha*(entrance->level+1))
				if(d_r<d_q*alpha)
				{
					is_good = false;
					break;
				}
				/*
				for(auto *pv : neighbourhood(*e.u,level))
					if(pv==e.u)
					{
						is_good = false;
						break;
					}
				*/
				/*
				if(nbh.find(e.u)!=nbh.end())
					is_good = false;
				*/
			}

			if(is_good)
			{
				R.push_back(e.u);
				/*				
				for(auto *pv : neighbourhood(*e.u,level))
					nbh.insert(pv);
				*/
			}
			else
				W_d.push_back(e);
		}

		// std::sort(W_d.begin(), W_d.end(), nearest());
		auto it = W_d.begin();
		// std::priority_queue<dist,parlay::sequence<dist>,farthest> res;
		auto &res = R;
		/*
		for(const auto &r : R)
		{
			res.push({U::distance(u,get_node(r).data,dim), r});
		}
		*/
		if(keepPrunedConnections)
		{
			// while(W_d.size()>0 && res.size()<M)
				// res.push(W_d.top()), W_d.pop();
			while(it!=W_d.end() && res.size()<M)
				// res.push(*(it++));
				res.push_back((it++)->u);
		}
		return ityr::global_vector<node_id>(res.begin(), res.end());
	}

	auto select_neighbors(const T &u, 
		/*const std::priority_queue<dist,parlay::sequence<dist>,farthest> &C,*/
		const std::vector<dist> &C, uint32_t M,
		uint32_t level, bool extendCandidate=false, bool keepPrunedConnections=false)
	{
		/*
		(void)level, (void)extendCandidate, (void)keepPrunedConnections;
		return select_neighbors_simple(u,C,M);
		*/
		return select_neighbors_heuristic(u, C, M, level, extendCandidate, keepPrunedConnections);
	}

        template <typename Rng>
	uint32_t get_level_random(Rng&& rng)
	{
		// static thread_local int32_t anchor;
		// uint32_t esp;
		// asm volatile("movl %0, %%esp":"=a"(esp));
		// static thread_local std::hash<std::thread::id> h;
		// static thread_local std::mt19937 gen{h(std::this_thread::get_id())};
		static std::uniform_real_distribution<> dis(std::numeric_limits<float>::min(), 1.0);
		const uint32_t res = uint32_t(-log(dis(rng))*m_l);
		return res;
	}

	// auto search_layer(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, uint64_t verbose=0) const; // To static
	auto search_layer(const node &u, const ityr::global_vector<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl={}) const; // To static
#if 0
	auto search_layer_new_ex(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl={}) const; // To static
	auto beam_search_ex(const node &u, const parlay::sequence<node_id> &eps, uint32_t beamSize, uint32_t l_c, search_control ctrl={}) const;
#endif
	auto get_threshold_m(uint32_t level){
		return level==0? m*2: m;
	}

#if 0
	void fix_edge()
	{
		fprintf(stderr, "Start fixing edges...\n");

		for(int32_t l_c=get_node(entrance[0]).level; l_c>=0; --l_c)
		{
			parlay::sequence<parlay::sequence<std::pair<node_id,node_id>>> edge_add(n);

			parlay::parallel_for(0, n, [&](uint32_t i){
				auto &u = node_pool[i]; // TODO: to fix
				if((uint32_t)l_c>u.level) return;

				auto &edge_v = edge_add[i];
				edge_v.clear();
				for(node_id pv : neighbourhood(u,l_c))
				{
					const auto &nbh_v = neighbourhood(get_node(pv),l_c);
					if(std::find_if(nbh_v.begin(),nbh_v.end(),[&](const node_id pu_extant){ // TODO: to optimize
						return pu_extant==i;
					})==nbh_v.end())
						edge_v.emplace_back(pv, i);
				}
			});

			auto edge_add_flatten = parlay::flatten(edge_add);
			auto edge_add_grouped = parlay::group_by_key(edge_add_flatten);

			parlay::sequence<uint32_t> indeg(n);
			parlay::parallel_for(0, n, [&](uint32_t i){
				indeg[i] = 0;
			});
			parlay::parallel_for(0, edge_add_grouped.size(), [&](size_t j){
				node_id pv = edge_add_grouped[j].first;
				indeg[U::get_id(get_node(pv).data)] = edge_add_grouped[j].second.size();
			});
			parlay::parallel_for(0, edge_add_grouped.size(), [&](size_t j){
				node_id pv = edge_add_grouped[j].first;
				auto &nbh_v = neighbourhood(get_node(pv),l_c);
				auto &nbh_v_add = edge_add_grouped[j].second;

				if(nbh_v.size()+nbh_v_add.size()>get_threshold_m(l_c))
				{
					parlay::sequence<dist> candidates;
					for(node_id pu : nbh_v)
						candidates.push_back({U::distance(get_node(pu).data,get_node(pv).data,dim), pu});
					for(node_id pu : nbh_v_add)
						candidates.push_back({U::distance(get_node(pu).data,get_node(pv).data,dim), pu});
					auto res = select_neighbors(get_node(pv).data, candidates, get_threshold_m(l_c), l_c);

					nbh_v.clear();
					for(node_id pu : res)
						nbh_v.push_back(pu);
				}
				else nbh_v.insert(nbh_v.end(),nbh_v_add.begin(), nbh_v_add.end());
			});
		}
	}
#endif

public:
#if 0
	auto get_deg(uint32_t level=0)
	{
		parlay::sequence<uint32_t> res;
		res.reserve(node_pool.size());
		for(const node &e : node_pool)
		{
			if(e.level>=level)
				res.push_back(e.neighbors[level].size());
		}
		return res;
	}

	auto get_indeg(uint32_t level) const
	{
		static uint32_t *indeg[16] = {nullptr};
		auto *&res = indeg[level];
		if(!res)
		{
			res = new uint32_t[n];
			for(uint32_t i=0; i<n; ++i)
				res[i] = 0;
			for(const node_id pu : node_pool)
			{
				if(get_node(pu).level<level) continue;
				for(const node_id pv : get_node(pu).neighbors[level])
					res[U::get_id(get_node(pv).data)]++;
			}
		}
		return res;
	}
#endif

	uint32_t get_height() const
	{
          auto e_cs = ityr::make_checkout(&node_pool[entrance[0].get()], 1, ityr::checkout_mode::read);
		return e_cs[0].level;
	}

	size_t cnt_degree(uint32_t l) const
	{
          return ityr::transform_reduce(
              ityr::execution::parallel_policy(1024),
              node_pool.begin(), node_pool.end(),
              ityr::reducer::plus<std::size_t>{},
              [=](const node& u) -> std::size_t {
                if (u.level < l) {
                  return 0;
                } else {
                  auto nbh_v_cs = ityr::make_checkout(&u.neighbors[l], 1, ityr::checkout_mode::read);
                  return nbh_v_cs[0].size();
                }
              });
	}

	size_t cnt_vertex(uint32_t l) const
	{
          return ityr::transform_reduce(
              ityr::execution::parallel_policy(1024),
              node_pool.begin(), node_pool.end(),
              ityr::reducer::plus<std::size_t>{},
              [=](const node& u) {
			return u.level<l? 0: 1;
              });
	}

	size_t get_degree_max(uint32_t l) const
	{
          return ityr::transform_reduce(
              ityr::execution::parallel_policy(1024),
              node_pool.begin(), node_pool.end(),
              ityr::reducer::max<std::size_t>{},
              [=](const node& u) -> std::size_t {
                if (u.level < l) {
                  return 0;
                } else {
                  auto nbh_v_cs = ityr::make_checkout(&u.neighbors[l], 1, ityr::checkout_mode::read);
                  return nbh_v_cs[0].size();
                }
              });
	}
/*
	void debug_output_graph(uint32_t l)
	{
		// return;
		debug_output("Printing the graph at level %u\n", l);
		auto node_exist = parlay::pack(
			node_pool,
			parlay::delayed_seq<bool>(node_pool.size(),[&](size_t i){
				return node_pool[i]->level>=l;
			})
		);
		const auto num_vertices = node_exist.size();
		const auto num_edges = parlay::reduce(
			parlay::delayed_seq<uint64_t>(node_exist.size(),[&](size_t i){
				return node_exist[i]->neighbors[l].size();
			}),
			parlay::addm<uint64_t>{}
		);
		debug_output("# vertices: %lu, # edges: %llu\n", num_vertices, num_edges);

		for(node_id pu : node_exist)
		{
			debug_output("node_id: %u\n", U::get_id(get_node(pu).data));
			// if(!res[i]) continue;
			debug_output("\tneighbors:");
			for(node_id pv : neighbourhood(get_node(pu),l))
				debug_output(" %u", U::get_id(get_node(pv).data));
			debug_output("\n");
		}
	}
*/
};

#if 0
template<typename U, template<typename> class Allocator>
template<typename G>
HNSW<U,Allocator>::HNSW(const std::string &filename_model, G getter)
{
	std::ifstream model(filename_model, std::ios::binary);
	if(!model.is_open())
		throw std::runtime_error("Failed to open the model");

	const auto size_buffer = 1024*1024*1024; // 1G
	auto buffer = std::make_unique<char[]>(size_buffer);
	model.rdbuf()->pubsetbuf(buffer.get(), size_buffer);

	auto read = [&](auto &data, auto ...args){
		auto read_impl = [&](auto &f, auto &data, auto ...args){
			using T = std::remove_reference_t<decltype(data)>;
			if constexpr(std::is_pointer_v<std::decay_t<T>>)
			{
				auto read_array = [&](auto &data, size_t size, auto ...args){
					for(size_t i=0; i<size; ++i)
						f(f, data[i], args...);
				};
				// use the array extent as the size
				if constexpr(sizeof...(args)==0 && std::is_array_v<T>)
				{
					read_array(data, std::extent_v<T>);
				}
				else
				{
					static_assert(sizeof...(args), "size was not provided");
					read_array(data, args...);
				}
			}
			else
			{
				static_assert(std::is_standard_layout_v<T>);
				model.read((char*)&data, sizeof(data));
			}
		};
		read_impl(read_impl, data, args...);
	};

	char model_type[5] = {'\000'};
	read(model_type, 4);
	if(strcmp(model_type,"HNSW"))
		throw std::runtime_error("Wrong type of model");
	uint32_t version;
	read(version);
	if(version!=3)
		throw std::runtime_error("Unsupported version");

	size_t code_U, size_node;
	read(code_U);
	read(size_node);
	if((typeid(U).hash_code()^sizeof(U))!=code_U)
		throw std::runtime_error("Inconsistent type `U`");
	if(sizeof(node)!=size_node)
		throw std::runtime_error("Inconsistent type `node`");

	// read parameter configuration
	read(dim);
	read(m_l);
	read(m);
	read(ef_construction);
	read(alpha);
	read(n);
	puts("Configuration loaded");
	printf("dim = %u\n", dim);
	printf("m_l = %f\n", m_l);
	printf("m = %u\n", m);
	printf("efc = %u\n", ef_construction);
	printf("alpha = %f\n", alpha);
	printf("n = %u\n", n);
	// read indices
	// std::unordered_map<uint32_t,node*> addr;
	node_pool.resize(n);
	for(uint32_t i=0; i<n; ++i)
	{
		// auto *u = new node;
		node &u = get_node(i);
		read(u.level);
		uint32_t id_u; // TODO: use generic type
		read(id_u);
		u.data = getter(id_u);
		// addr[id_u] = u;
	}
	for(node &u : node_pool)
	{
		u.neighbors = new parlay::sequence<node_id>[u.level+1];
		for(uint32_t l=0; l<=u.level; ++l)
		{
			size_t size;
			read(size);
			auto &nbh_u = u.neighbors[l];
			nbh_u.reserve(size);
			for(size_t i=0; i<size; ++i)
			{
				uint32_t id_v;
				read(id_v);
				nbh_u.push_back(id_v);
			}
		}
	}
	// read entrances
	size_t size;
	read(size);
	entrance.reserve(size);
	for(size_t i=0; i<size; ++i)
	{
		uint32_t id_u;
		read(id_u);
		entrance.push_back(id_u);
	}
}
#endif

template<typename U, template<typename> class Allocator>
template<typename Iter>
HNSW<U,Allocator>::HNSW(Iter begin, Iter end, uint32_t dim_, float m_l_, uint32_t m_, uint32_t ef_construction_, float alpha_, float batch_base, bool do_fixing [[maybe_unused]])
	: entrance(ityr::global_vector_options(true, 1024)), // coll
          dim(dim_), m_l(m_l_), m(m_), ef_construction(ef_construction_), alpha(alpha_), n(std::distance(begin,end)),
          node_pool(ityr::global_vector_options(true, 1024)) // coll
{
	static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, T>);
	static_assert(std::is_base_of_v<
		std::random_access_iterator_tag, typename std::iterator_traits<Iter>::iterator_category>);

	if(n==0) return;

        /* ityr::root_exec([=] { */
        /*   uint64_t seed = 42; */
        /*   ityr::default_random_engine rng(seed); */
        /*   ityr::shuffle(ityr::execution::parallel_policy(1024 * 16), */
        /*                 begin, end, rng); */
        /* }); */

        uint64_t seed = 42;
        ityr::default_random_engine rng(seed);

	const auto level_ep = get_level_random(rng);
	// node *entrance_init = allocator.allocate(1);
	// node_pool.push_back(entrance_init);
	node_id entrance_init = 0;
        node_pool.reserve(n);
        node_pool.emplace_back(level_ep, ityr::global_vector<ityr::global_vector<node_id>>(level_ep+1), (*begin).get()/*anything else*/);
	entrance.push_back(entrance_init);

        ityr::profiler_begin();

	uint32_t batch_begin=0, batch_end=1, size_limit=n*0.02;
	float progress = 0.0;
	while(batch_end<n)
	{
		batch_begin = batch_end;
		batch_end = std::min({n, (uint32_t)std::ceil(batch_begin*batch_base)+1, batch_begin+size_limit});
		/*
		if(batch_end>batch_begin+100)
			batch_end = batch_begin+100;
		*/
		// batch_end = batch_begin+1;

		insert(begin+batch_begin, begin+batch_end, true, rng);
		// insert(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, false);

		if(batch_end>n*(progress+0.05))
		{
			progress = float(batch_end)/n;
                        my_printf("Built: %3.2f%%\n", progress*100);
			my_printf("# visited: %lu\n", ityr::common::mpi_reduce_value(total_visited, 0, MPI_COMM_WORLD));
			my_printf("# eval: %lu\n", ityr::common::mpi_reduce_value(total_eval, 0, MPI_COMM_WORLD));
			my_printf("size of C: %lu\n", ityr::common::mpi_reduce_value(total_size_C, 0, MPI_COMM_WORLD));

                        ityr::profiler_end();
                        ityr::profiler_flush();
                        ityr::profiler_begin();
		}
	}

        my_printf("# visited: %lu\n", ityr::common::mpi_reduce_value(total_visited, 0, MPI_COMM_WORLD));
        my_printf("# eval: %lu\n", ityr::common::mpi_reduce_value(total_eval, 0, MPI_COMM_WORLD));
        my_printf("size of C: %lu\n", ityr::common::mpi_reduce_value(total_size_C, 0, MPI_COMM_WORLD));

        ityr::profiler_end();
        ityr::profiler_flush();
#if 0
	if(do_fixing) fix_edge();
#endif

	#if 0
		for(const auto *pu : node_pool)
		{
			fprintf(stderr, "[%u] (%.2f,%.2f)\n", U::get_id(get_node(pu).data), get_node(pu).data[0], get_node(pu).data[1]);
			for(int32_t l=pu->level; l>=0; --l)
			{
				fprintf(stderr, "\tlv. %d:", l);
				for(const auto *k : pu->neighbors[l])
					fprintf(stderr, " %u", U::get_id(get_node(k).data));
				fputs("\n", stderr);
			}
		}
	#endif
/*
	for(uint32_t l=0; l<entrance[0]->level; ++l)
		debug_output_graph(l);
*/
}

template<typename U, template<typename> class Allocator>
template<typename Iter, typename Rng>
void HNSW<U,Allocator>::insert(Iter begin, Iter end, bool from_blank, Rng&& rng)
{
      ityr::root_exec([=, rng = rng.split()]() mutable {
        timer t;

        ityr::global_vector_options global_vec_coll_opts(true, 1024);

        auto e_cs = ityr::make_checkout(&node_pool[entrance[0].get()], 1, ityr::checkout_mode::read);
	const auto level_ep = e_cs[0].level;
        e_cs.checkin();
	const auto size_batch = std::distance(begin,end);
        ityr::global_vector<ityr::global_vector<node_id>> eps(global_vec_coll_opts, size_batch);
	//const float factor_m = from_blank? 0.5: 1;
	const auto factor_m = 1;

        auto offset = node_pool.size();

	debug_output("Insert %lu elements; from blank? [%c]\n", size_batch, "NY"[from_blank]);

	// auto *pool = allocator.allocate(size_batch);
	// first, query the nearest point as the starting point for each node to insert
	if(from_blank)
	{
          ityr::coll_exec([=] {
            node_pool.resize(offset+size_batch);
          });
          ityr::transform(
              ityr::execution::parallel_policy(1024),
              begin, end, ityr::count_iterator<uint64_t>(0), node_pool.begin() + offset,
              [=](const T& q, uint64_t i) {
                auto rng_ = rng;
                const auto level_u = get_level_random(rng_.split(i));
                return node{level_u, ityr::global_vector<ityr::global_vector<node_id>>(level_u + 1), q};
              });
	}
	else
	{
#if 0
	parlay::parallel_for(0, size_batch, [&](uint32_t i){
		node_new[i] = node_pool.size()-size_batch+i;
	});
#else
        throw std::runtime_error("from_blank false");
#endif
	}

        struct edge {
          node_id src;
          node_id dst;
        };

        ityr::global_vector<ityr::global_vector<edge>> edge_add(global_vec_coll_opts, size_batch);

        ityr::global_vector<std::size_t> edge_indices(global_vec_coll_opts, size_batch + 1, 0);

	debug_output("insert: prologue: %.4f\n", t.tick_s());
	debug_output("Nodes are settled\n");
	// TODO: merge ops
        ityr::transform(
            ityr::execution::par,
            node_pool.begin() + offset, node_pool.end(), eps.begin(),
            [=](const node& u) {
              const auto level_u = u.level;
              ityr::global_vector<node_id> eps_u(entrance.begin(), entrance.end());
              // eps_u.push_back(entrance);
              for(uint32_t l=level_ep; l>level_u; --l)
              {
                      const auto res = search_layer(u, eps_u, 1, l); // TODO: optimize
                      eps_u.clear();
                      eps_u.push_back(res[0].u);
              }
              return eps_u;
            });

	debug_output("insert: search entrance: %.4f\n", t.tick_s());
	debug_output("Finish searching entrances\n");
	// then we process them layer by layer (from high to low)
	for(int32_t l_c=level_ep; l_c>=0; --l_c) // TODO: fix the type
	{
		debug_output("Finding neighbors on lev. %d\n", l_c);
                static ityr::wallclock_t t_acc;
#if DEBUG_OUTPUT
                ityr::coll_exec([] { t_acc = 0; });
#endif
                ityr::for_each(
                    ityr::execution::par,
                    ityr::make_global_iterator(node_pool.begin() + offset, ityr::checkout_mode::read),
                    ityr::make_global_iterator(node_pool.end()           , ityr::checkout_mode::read),
                    ityr::make_global_iterator(eps.begin()               , ityr::checkout_mode::read_write),
                    ityr::make_global_iterator(edge_add.begin()          , ityr::checkout_mode::read_write),
                    ityr::count_iterator<node_id>(offset),
                    [=](node& u, auto& eps_u, auto& edge_u, node_id pu) {
                      if((uint32_t)l_c>u.level) return;

#if DEBUG_OUTPUT
                      timer t;
                      auto res = search_layer(u, eps_u, ef_construction, l_c);
                      t_acc += t.tick_ns();
#else
                      auto res = search_layer(u, eps_u, ef_construction, l_c);
#endif
                      auto neighbors_vec = select_neighbors(u.data, res, get_threshold_m(l_c)*factor_m, l_c);
                      // nbh_u.clear();
                      edge_u.clear();
                      // nbh_u.reserve(neighbors_vec.size());
                      edge_u.resize(neighbors_vec.size());
                      /*
                      for(uint32_t j=0; neighbors_vec.size()>0; ++j)
                      {
                              auto *pv = neighbors_vec.top().u;
                              neighbors_vec.pop();
                              // nbh_u[j] = pv;
                              // edge_u[j] = std::make_pair(pv, &u);
                              nbh_u.push_back(pv);
                              edge_u.emplace_back(pv, &u);
                      }
                      */
                      ityr::transform(
                          ityr::execution::sequenced_policy(neighbors_vec.size()),
                          neighbors_vec.begin(), neighbors_vec.end(), edge_u.begin(),
                          [=](node_id pv) { return edge{pv, pu}; });

                      if((uint32_t)l_c<=u.level) {
                        auto nbh_v_cs = ityr::make_checkout(&u.neighbors[l_c], 1, ityr::checkout_mode::read_write);
                        nbh_v_cs[0] = std::move(neighbors_vec);
                      }

                      eps_u.clear();
                      /*
                      while(res.size()>0)
                      {
                              eps_u.push_back(res.top().u); // TODO: optimize
                              res.pop();
                      }
                      */
                      eps_u.resize(res.size());
                      ityr::transform(
                          ityr::execution::sequenced_policy(res.size()),
                          res.begin(), res.end(), eps_u.begin(),
                          [=](const dist& e) { return e.u; });
                    });
                debug_output("insert: search_layer (level %d): %.4f\n", l_c, ityr::coll_exec([] {
                  return ityr::common::mpi_reduce_value(t_acc, 0, MPI_COMM_WORLD);
                }) / 1000000000.0 / ityr::n_ranks());

                debug_output("insert: add forward edges (level %d): %.4f\n", l_c, t.tick_s());
		debug_output("Adding reverse edges\n");
		// now we add edges in the other direction
		/* auto edge_add_flatten = parlay::flatten(edge_add); */
                ityr::global_vector<edge> edge_add_flatten(global_vec_coll_opts);

                ityr::transform_inclusive_scan(
                    ityr::execution::parallel_policy(1024),
                    edge_add.begin(), edge_add.end(), edge_indices.begin() + 1,
                    ityr::reducer::plus<std::size_t>{},
                    [](const auto& edge_u) { return edge_u.size(); });

                edge_add_flatten.resize(edge_indices.back().get());

                ityr::global_span<edge> edge_add_flatten_ref(edge_add_flatten);

                ityr::for_each(
                    ityr::execution::par,
                    ityr::make_global_iterator(edge_add.begin()    , ityr::checkout_mode::read),
                    ityr::make_global_iterator(edge_add.end()      , ityr::checkout_mode::read),
                    ityr::make_global_iterator(edge_indices.begin(), ityr::checkout_mode::read),
                    [=](auto& edge_u, std::size_t index_b) {
                      ityr::move(ityr::execution::sequenced_policy(1024),
                                 edge_u.begin(), edge_u.end(), edge_add_flatten_ref.begin() + index_b);
                    });

                debug_output("insert: flatten (level %d): %.4f\n", l_c, t.tick_s());

		/* auto edge_add_grouped = parlay::group_by_key(edge_add_flatten); */
                groupby(
                    ityr::make_global_iterator(edge_add_flatten.begin(), ityr::checkout_mode::read_write),
                    ityr::make_global_iterator(edge_add_flatten.end()  , ityr::checkout_mode::read_write),
                    [](const auto& a, const auto& b) { return a.src < b.src; },
                    [=](auto first, auto last) {
                      auto edges_cs = ityr::make_checkout(first, last - first, ityr::checkout_mode::read);
                      node_id pv = edges_cs[0].src;

                      auto node_cs = ityr::make_checkout(&node_pool[pv], 1, ityr::checkout_mode::read);
                      auto& u = node_cs[0];

                      auto nbh_v_cs = ityr::make_checkout(&u.neighbors[l_c], 1, ityr::checkout_mode::read_write);
                      auto &nbh_v = nbh_v_cs[0];

                      auto nbh_cs = ityr::make_checkout(nbh_v.data(), nbh_v.size(), ityr::checkout_mode::read);

                      std::vector<node_id> nbh_v_add;
                      nbh_v_add.reserve(edges_cs.size());
                      for (auto&& [_, pu] : edges_cs) {
                        bool is_extant = pu==pv||std::find_if(nbh_cs.begin(), nbh_cs.end(), [pu=pu](const node_id pu_extant){
                            return pu==pu_extant;
                            })!=nbh_cs.end();

                        if (!is_extant) {
                          nbh_v_add.push_back(pu);
                        }
                      }

                      const uint32_t size_nbh_total = nbh_cs.size()+nbh_v_add.size();

                      const auto m_s = get_threshold_m(l_c)*factor_m;
                      if(size_nbh_total>m_s)
                      {
                              auto candidates = std::vector<dist>(size_nbh_total);
                              for(size_t k=0; k<nbh_cs.size(); ++k) {
                                auto v_cs = ityr::make_checkout(&node_pool[nbh_cs[k]], 1, ityr::checkout_mode::read);
                                      candidates[k] = dist{U::distance(v_cs[0].data,u.data,dim), nbh_cs[k]};
                              }
                              for(size_t k=0; k<nbh_v_add.size(); ++k) {
                                auto v_cs = ityr::make_checkout(&node_pool[nbh_v_add[k]], 1, ityr::checkout_mode::read);
                                      candidates[k+nbh_cs.size()] = dist{U::distance(v_cs[0].data,u.data,dim), nbh_v_add[k]};
                              }

                              std::sort(candidates.begin(), candidates.end(), farthest());

                              nbh_cs.checkin();

                              nbh_v.resize(m_s);
                              ityr::transform(
                                  ityr::execution::sequenced_policy(1024),
                                  candidates.begin(), candidates.begin() + m_s,
                                  nbh_v.begin(),
                                  [](const auto& c) { return c.u; });
                              /*
                              auto res = select_neighbors(get_node(pv).data, candidates, m_s, l_c);
                              nbh_v.clear();
                              for(auto *pu : res)
                                      nbh_v.push_back(pu);
                              */
                              // nbh_v = select_neighbors(get_node(pv).data, candidates, m_s, l_c);
                      }
                      else {
                        nbh_cs.checkin();
                        nbh_v.insert(nbh_v.end(),nbh_v_add.begin(), nbh_v_add.end());
                      }
                    });

                debug_output("insert: groupby (level %d): %.4f\n", l_c, t.tick_s());
	}

	debug_output("Updating entrance\n");
	// finally, update the entrance
	auto node_highest_gptr = ityr::max_element(
            ityr::execution::parallel_policy(1024),
            node_pool.begin() + offset, node_pool.end(),
            [&](const node& u, const node& v){
              return u.level < v.level;
            });
        node_id node_highest = node_highest_gptr - node_pool.begin();
        auto node_cs = ityr::make_checkout(node_highest_gptr, 1, ityr::checkout_mode::read);
        auto node_level = node_cs[0].level;
        auto node_data = node_cs[0].data;
        node_cs.checkin();
	if(node_level>level_ep)
	{
          ityr::coll_exec([=] {
		entrance.clear();
		entrance.push_back(node_highest);
                });
		debug_output("New entrance [%u] at lev %u\n", U::get_id(node_data), node_level);
	}
	else if(node_level==level_ep)
	{
          ityr::coll_exec([=] {
		entrance.push_back(node_highest);
                });
		debug_output("New entrance [%u] at lev %u\n", U::get_id(node_data), node_level);
	}

        debug_output("insert: update entrance: %.4f\n", t.tick_s());

	// and add new nodes to the pool
	/*
	if(from_blank)
	node_pool.insert(node_pool.end(), node_new.get(), node_new.get()+size_batch);
	*/

        debug_output("insert: total: %.4f\n", t.total_duration_s());
      });
}

template<typename U, template<typename> class Allocator>
auto HNSW<U,Allocator>::search_layer(const node &u, const ityr::global_vector<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl) const
{
	// parlay::sequence<bool> visited(n);
	//parlay::sequence<uint32_t> visited(mask+1, n+1);
	// TODO: Try hash to an array
	// TODO: monitor the size of `visited`
	std::unordered_set<node_id> visited;
        std::vector<dist> W, discarded;
	std::set<dist,farthest> C;
	W.reserve(ef);

        {
          auto eps_cs = ityr::make_checkout(eps.data(), eps.size(), ityr::checkout_mode::read);
          for(node_id ep : eps_cs)
          {
            auto e_cs = ityr::make_checkout(&node_pool[ep], 1, ityr::checkout_mode::read);
                  //const auto id = U::get_id(get_node(ep).data);
                  //visited[parlay::hash64_2(id)&mask] = id;
                  visited.insert(ep);
                  const auto d = U::distance(u.data,e_cs[0].data,dim);
                  C.insert({d,ep});
                  W.push_back({d,ep});
          }
        }
        // std::make_heap(C.begin(), C.end(), nearest());
        std::make_heap(W.begin(), W.end(), farthest());

	uint32_t cnt_eval = 0;
	uint32_t limit_eval = ctrl.limit_eval.value_or(n);
	while(C.size()>0)
	{
		if(ctrl.skip_search) break;
		if(C.begin()->d>W[0].d*ctrl.beta) break;

		if(++cnt_eval>limit_eval) break;
#if 0
		if(ctrl.log_dist)
		{
			std::array<float,5> t;

			if(ctrl.log_size)
			{
				t[0] = W[0].d;
				t[1] = W.size();
				t[2] = C.size();
				vc_in_search[*ctrl.log_size].push_back(t);
			}

			auto it = C.begin();
			const auto step = C.size()/4;
			for(uint32_t i=0; i<4; ++i)
				t[i]=it->d, std::advance(it,step);
			t[4] = C.rbegin()->d;

			dist_in_search[*ctrl.log_dist].push_back(t);
		}
#endif

                auto c_cs = ityr::make_checkout(&node_pool[C.begin()->u], 1, ityr::checkout_mode::read);
		const auto &c = c_cs[0];
		// std::pop_heap(C.begin(), C.end(), nearest());
		// C.pop_back();
		C.erase(C.begin());

                auto nbh_v_cs = ityr::make_checkout(&c.neighbors[l_c], 1, ityr::checkout_mode::read);
                auto &nbh_v = nbh_v_cs[0];

                auto nbh_cs = ityr::make_checkout(nbh_v.data(), nbh_v.size(), ityr::checkout_mode::read);

		for(node_id pv: nbh_cs)
		{
			//const auto id = U::get_id(get_node(pv).data);
			//const auto idx = parlay::hash64_2(id)&mask;
			//if(visited[idx]==id) continue;
			//visited[idx] = id;
			if(!visited.insert(pv).second) continue;
                        auto node_cs = ityr::make_checkout(&node_pool[pv], 1, ityr::checkout_mode::read);
                        auto& node = node_cs[0];
			const auto d = U::distance(u.data,node.data,dim);
			if(W.size()<ef||d<W[0].d)
			{
				C.insert({d,pv});

				// C.push_back({d,pv,dc+1});
				// std::push_heap(C.begin(), C.end(), nearest());
				W.push_back({d,pv});
                                std::push_heap(W.begin(), W.end(), farthest());
                                if(W.size()>ef)
                                {
                                        std::pop_heap(W.begin(), W.end(), farthest());
                                        if(ctrl.radius && W.back().d<=*ctrl.radius)
                                                discarded.push_back(W.back());
                                        W.pop_back();
                                }
				if(C.size()>ef)
					C.erase(std::prev(C.end()));
			}
		}
	}

	//total_visited += visited.size();
	//total_visited += visited.size()-std::count(visited.begin(),visited.end(),n+1);
	total_visited += visited.size();
	total_size_C += C.size()+cnt_eval;
	total_eval += cnt_eval;

	if(ctrl.log_per_stat)
	{
		const auto qid = *ctrl.log_per_stat;
                auto [pv, pe, ps] =
                  ityr::make_checkouts(&per_visited[qid], 1, ityr::checkout_mode::read_write,
                                       &per_eval[qid]   , 1, ityr::checkout_mode::read_write,
                                       &per_size_C[qid] , 1, ityr::checkout_mode::read_write);
		pv[0] += visited.size();
		pe[0] += C.size()+cnt_eval;
		ps[0] += cnt_eval;
	}

#if 0
	if(ctrl.radius)
	{
		const auto rad = *ctrl.radius;
		auto split = std::partition(W.begin(), W.end(), [rad](const dist &e){
			return e.d <= rad;
		});
		W.resize(split-W.begin());
		W.append(discarded);
		total_range_candidate[parlay::worker_id()] += W.size();
	}
#endif
	return W;
}

#if 0
template<typename U, template<typename> class Allocator>
auto HNSW<U,Allocator>::search_layer_new_ex(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl) const
{
	auto verbose_output = [&](const char *fmt, ...){
		if(!ctrl.verbose_output) return;

		va_list args;
		va_start(args, fmt);
		vfprintf(stderr, fmt, args);
		va_end(args);
	};

	parlay::sequence<std::array<float,5>> dummy;
	auto &dist_range = ctrl.log_dist? dist_in_search[*ctrl.log_dist]: dummy;
	uint32_t cnt_eval = 0;

	auto *indeg = ctrl.verbose_output? get_indeg(l_c): reinterpret_cast<const uint32_t*>(node_pool.data());
	// parlay::sequence<bool> visited(n);
	// TODO: Try hash to an array
	// TODO: monitor the size of `visited`
	std::set<uint32_t> visited;
	// std::priority_queue<dist_ex,parlay::sequence<dist_ex>,nearest> C;
	// std::priority_queue<dist_ex,parlay::sequence<dist_ex>,farthest> W;
	parlay::sequence<dist_ex> /*C, W, */W_;
	std::set<dist_ex,farthest> C, C_acc;
	uint32_t cnt_used = 0;

	for(node_id ep : eps)
	{
		// visited[U::get_id(get_node(ep).data)] = true;
		const auto id = U::get_id(get_node(ep).data);
		visited.insert(id);
		const auto d = U::distance(u.data,get_node(ep).data,dim);
		C.insert({d,ep,1});
		C_acc.insert({d,ep,1});
		// C.push_back({d,ep,1});
		// W.push_back({d,ep,1});
		verbose_output("Insert\t[%u](%f) initially\n", id, d);
	}
	// std::make_heap(C.begin(), C.end(), nearest());
	// std::make_heap(W.begin(), W.end(), farthest());

	// static thread_local std::mt19937 gen{parlay::worker_id()};
	// static thread_local std::exponential_distribution<float> distro{48};
	while(C.size()>0)
	{
		// const auto &f = *(W[0].u);
		// if(U::distance(c.data,u.data,dim)>U::distance(f.data,u.data,dim))
		// if(C[0].d>W[0].d) break;
		if(C_acc.size()==cnt_used) break;
		cnt_eval++;

		if(ctrl.log_dist)
			dist_range.push_back({C.begin()->d,C.rbegin()->d});
		/*
		const auto dc = C[0].depth;
		const auto &c = *(C[0].u);
		*/
		auto it = C.begin();
		/*
		float quantile = distro(gen);
		if(quantile>C.size())
			quantile = C.size();
		const auto dis_min = C.begin()->d;
		const auto dis_max = C.rbegin()->d;
		const auto threshold = quantile/C.size()*(dis_max-dis_min) + dis_min - 1e-6;
		auto it = C.lower_bound(dist_ex{threshold,nullptr,0});
		*/
		const auto dc = it->depth;
		const auto &c = *(it->u);
		// W_.push_back(C[0]);
		W_.push_back(*it);
		// std::pop_heap(C.begin(), C.end(), nearest());
		// C.pop_back();
		C.erase(it);
		cnt_used++;

		verbose_output("------------------------------------\n");
		const uint32_t id_c = U::get_id(c.data);
		verbose_output("Eval\t[%u](%f){%u}\t[%u]\n", id_c, it->d, dc, indeg[id_c]);
		uint32_t cnt_insert = 0;
		for(node_id pv: neighbourhood(c, l_c))
		{
			// if(visited[U::get_id(get_node(pv).data)]) continue;
			// visited[U::get_id(get_node(pv).data)] = true;
			if(!visited.insert(U::get_id(get_node(pv).data)).second) continue;
			// const auto &f = *(W[0].u);
			// if(W.size()<ef||U::distance(get_node(pv).data,u.data,dim)<U::distance(f.data,u.data,dim))
			const auto d = U::distance(u.data,get_node(pv).data,dim);
			// if(W.size()<ef||d<W[0].d)
			// if(C.size()<ef||d<C.rend()->d)
			{
				// C.push_back({d,pv,dc+1});
				// std::push_heap(C.begin(), C.end(), nearest());
				/*
				W.push_back({d,pv,dc+1});
				std::push_heap(W.begin(), W.end(), farthest());
				if(W.size()>ef)
				{
					std::pop_heap(W.begin(), W.end(), farthest());
					W.pop_back();
				}
				*/
				if(C.size()<ef || d<C.rbegin()->d)
				{
				C.insert({d,pv,dc+1});
				const uint32_t id_v = U::get_id(get_node(pv).data);
				verbose_output("Insert\t[%u](%f){%u}\t[%u](%f)\n", 
					id_v, d, dc+1, 
					indeg[id_v], U::distance(c.data,get_node(pv).data,dim)
				);
				cnt_insert++;
				if(C.size()>ef)
				{
					// std::pop_heap(C.begin(), C.end(), nearest());
					// C.pop_back();
					C.erase(std::prev(C.end()));
				}
				}
				if(C_acc.size()<ef || d<C_acc.rbegin()->d)
				{
				C_acc.insert({d,pv,dc+1});
				if(C_acc.size()>ef)
				{
					auto it = std::prev(C_acc.end());
					if(std::find_if(W_.begin(), W_.end(), [&](const dist_ex &a){
						return a.u==it->u;
					})!=W_.end())
						cnt_used--;
					C_acc.erase(it);
				}
				}
			}
		}
		verbose_output("%u inserts in this round\n", cnt_insert);
	}
	if(l_c==0)
	{
		const auto id = parlay::worker_id();
		total_visited[id] += visited.size();
		total_size_C[id] += C.size()+cnt_eval;
		total_eval[id] += cnt_eval;
	}
	/*
	std::sort(W.begin(), W.end(), farthest());
	if(W.size()>ef) W.resize(ef);
	*/
	return W_;
}

template<typename U, template<typename> class Allocator>
auto HNSW<U,Allocator>::beam_search_ex(const node &u, const parlay::sequence<node_id> &eps, uint32_t beamSize, uint32_t l_c, search_control ctrl) const
// std::pair<parlay::sequence<dist_ex>, parlay::sequence<dist_ex>> beam_search(
		// T* p_coords, int beamSize)
{
	// beamSize *= 2;
	// beamSize = 20000;
	// initialize data structures
	parlay::sequence<dist_ex> visited;
	parlay::sequence<dist_ex> frontier;
	auto dist_less = [&](const dist_ex &a, const dist_ex &b) {
		return a.d < b.d || (a.d == b.d && a.u < b.u);
		// return a.u<b.u;
	};
	auto dist_eq = [&](const dist_ex &a, const dist_ex &b){
		return a.u == b.u;
	};

	// int bits = std::ceil(std::log2(beamSize * beamSize));
	// parlay::sequence<uint32_t> hash_table(1 << bits, std::numeric_limits<uint32_t>::max());
	std::set<uint32_t> accessed;

	auto make_pid = [&] (node_id ep) {
		const auto d = U::distance(u.data,get_node(ep).data,dim);
		return dist_ex{d,ep,1};
	};

	// the frontier starts with the medoid
	// frontier.push_back(make_pid(medoid->id));
	
	for(node_id ep : eps)
		frontier.push_back(make_pid(ep));
	std::sort(frontier.begin(), frontier.end(), dist_less);
	
	// frontier.push_back(make_pid(eps[0]));

	parlay::sequence<dist_ex> unvisited_frontier;
	// parlay::sequence<dist_ex> unvisited_frontier(beamSize);
	parlay::sequence<dist_ex> new_frontier;
	// parlay::sequence<dist_ex> new_frontier(2 * beamSize);
	bool not_done = true;


	for(size_t i=0; i<frontier.size(); ++i)
	{
		unvisited_frontier.push_back(frontier[i]);
		// unvisited_frontier[i] = frontier[i];
		accessed.insert(U::get_id(frontier[i].get_node(u).data));
	}

	// terminate beam search when the entire frontier has been visited
	while (not_done) {
		// the next node to visit is the unvisited frontier node that is closest
		// to p
		dist_ex currentPid = unvisited_frontier[0];
		node_id current_vtx = currentPid.u;
		debug_output("current_vtx ID: %u\n", U::get_id(get_node(current_vtx).data));

		auto g = [&](node_id a) {
			uint32_t id_a = U::get_id(get_node(a).data);
			/*
			uint32_t loc = parlay::hash64_2(id_a) & ((1 << bits) - 1);
			if (hash_table[loc] == id_a) return false;
			hash_table[loc] = id_a;
			return true;
			*/
			return accessed.insert(id_a).second;
		};

		parlay::sequence<node_id> candidates;
		auto f = [&](node_id pu, node_id pv/*, empty_weight wgh*/) {
			if (g(pv)) {
				candidates.push_back(pv);
			}
			return true;
		};
		for(node_id pv : neighbourhood(get_node(current_vtx),l_c))
			// current_vtx.out_neighbors().foreach_cond(f);
			f(current_vtx, pv);

		debug_output("candidates:\n");
		for(node_id p : candidates)
			debug_output("%u ", U::get_id(get_node(p).data));
		debug_output("\n");
		auto pairCandidates =
				parlay::map(candidates, make_pid);
		/*
		auto sortedCandidates =
				parlay::unique(parlay::sort(pairCandidates, dist_less), dist_eq);
		*/
		auto &sortedCandidates = pairCandidates;
		debug_output("size of sortedCandidates: %lu\n", sortedCandidates.size());
		/*
		auto f_iter = std::set_union(
				frontier.begin(), frontier.end(), sortedCandidates.begin(),
				sortedCandidates.end(), new_frontier.begin(), dist_less);\
		*/
		sortedCandidates.insert(sortedCandidates.end(), frontier);
		new_frontier = parlay::unique(parlay::sort(sortedCandidates,dist_less), dist_eq);

		// size_t f_size = std::min<size_t>(beamSize, f_iter - new_frontier.begin());
		size_t f_size = std::min<size_t>(beamSize, new_frontier.size());
		debug_output("f_size: %lu\n", f_size);

		debug_output("frontier (size: %lu)\n", frontier.size());
		for(const auto &e : frontier)
			debug_output("%u ", U::get_id(e.get_node(u).data));
		debug_output("\n");
		
		frontier =
				parlay::tabulate(f_size, [&](size_t i) { return new_frontier[i]; });
		debug_output("size of frontier: %lu\n", frontier.size());
		visited.insert(
				std::upper_bound(visited.begin(), visited.end(), currentPid, dist_less),
				currentPid);
		debug_output("size of visited: %lu\n", visited.size());
		unvisited_frontier.reserve(frontier.size());
		auto uf_iter =
				std::set_difference(frontier.begin(), frontier.end(), visited.begin(),
														visited.end(), unvisited_frontier.begin(), dist_less);
		debug_output("uf_iter - unvisited_frontier.begin(): %lu\n", uf_iter - unvisited_frontier.begin());
		not_done = uf_iter > unvisited_frontier.begin();

		if(l_c==0)
			total_visited[parlay::worker_id()] += candidates.size();
	}
	parlay::sequence<dist_ex> W;
	W.insert(W.end(), visited);
	return W;
}
#endif
/*
template<typename U, template<typename> class Allocator>
parlay::sequence<std::pair<uint32_t,float>> HNSW<U,Allocator>::search(const T &q, uint32_t k, uint32_t ef, search_control ctrl)
{
	auto res_ex = search_ex(q,k,ef,ctrl);
	parlay::sequence<std::pair<uint32_t,float>> res;
	res.reserve(res_ex.size());
	for(const auto &e : res_ex)
		res.emplace_back(std::get<0>(e), std::get<2>(e));

	return res;
}
*/
template<typename U, template<typename> class Allocator>
ityr::global_vector<std::pair<uint32_t,float>> HNSW<U,Allocator>::search(const T &q, uint32_t k, uint32_t ef, search_control ctrl)
{
	total_range_candidate = 0;
	total_visited = 0;
	total_eval = 0;
	total_size_C = 0;

	if(ctrl.log_per_stat)
	{
		const auto qid = *ctrl.log_per_stat;
                auto [pv, pe, ps] =
                  ityr::make_checkouts(&per_visited[qid], 1, ityr::checkout_mode::write,
                                       &per_eval[qid]   , 1, ityr::checkout_mode::write,
                                       &per_size_C[qid] , 1, ityr::checkout_mode::write);
		pv[0] = 0;
		pe[0] = 0;
		ps[0] = 0;
	}

	node u{0, {}, q}; // To optimize
	// std::priority_queue<dist,parlay::sequence<dist>,farthest> W;
        ityr::global_vector<node_id> eps(entrance.begin(), entrance.end());
	if(!ctrl.indicate_ep) {
          auto e_cs = ityr::make_checkout(&node_pool[entrance[0].get()], 1, ityr::checkout_mode::read);
		for(int l_c=e_cs[0].level; l_c>0; --l_c) // TODO: fix the type
		{
			search_control c{};
			c.log_per_stat = ctrl.log_per_stat; // whether count dist calculations at all layers
			// c.limit_eval = ctrl.limit_eval; // whether apply the limit to all layers
			const auto W = search_layer(u, eps, 1, l_c, c);
			eps.clear();
			eps.push_back(W[0].u);
			/*
			while(!W.empty())
			{
				eps.push_back(W.top().u);
				W.pop();
			}
			*/
		}
        }
	else eps = {*ctrl.indicate_ep};
	auto W_ex = search_layer(u, eps, ef, 0, ctrl);
	// auto W_ex = search_layer_new_ex(u, eps, ef, 0, ctrl);
	// auto W_ex = beam_search_ex(u, eps, ef, 0);
	// auto R = select_neighbors_simple(q, W_ex, k);

	auto &R = W_ex;
	if(!ctrl.radius && R.size()>k) // the range search ignores the given k
	{
		std::sort(R.begin(), R.end(), farthest());
		if(k>0)
			k = std::upper_bound(R.begin()+k, R.end(), R[k-1], farthest())-R.begin();
		R.resize(k);
	}

        ityr::global_vector<std::pair<uint32_t,float>> res;
	res.reserve(R.size());
	/*
	while(W_ex.size()>0)
	{
		res.push_back({U::get_id(W_ex.top().get_node(u).data), W_ex.top().depth, W_ex.top().d});
		W_ex.pop();
	}
	*/
        for(const auto &e : R) {
          auto e_cs = ityr::make_checkout(&node_pool[e.u], 1, ityr::checkout_mode::read);
                res.push_back({U::get_id(e_cs[0].data),/* e.depth,*/ e.d});
        }
	return res;
}

#if 0
template<typename U, template<typename> class Allocator>
void HNSW<U,Allocator>::save(const std::string &filename_model) const
{
	std::ofstream model(filename_model, std::ios::binary);
	if(!model.is_open())
		throw std::runtime_error("Failed to create the model");

	const auto size_buffer = 1024*1024*1024; // 1G
	auto buffer = std::make_unique<char[]>(size_buffer);
	model.rdbuf()->pubsetbuf(buffer.get(), size_buffer);

	const auto write = [&](const auto &data, auto ...args){
		auto write_impl = [&](auto &f, const auto &data, auto ...args){
			using T = std::remove_reference_t<decltype(data)>;
			if constexpr(std::is_pointer_v<std::decay_t<T>>)
			{
				auto write_array = [&](const auto &data, size_t size, auto ...args){
					for(size_t i=0; i<size; ++i)
						f(f, data[i], args...);
				};
				// use the array extent as the size
				if constexpr(sizeof...(args)==0 && std::is_array_v<T>)
				{
					write_array(data, std::extent_v<T>);
				}
				else
				{
					static_assert(sizeof...(args), "size was not provided");
					write_array(data, args...);
				}
			}
			else
			{
				static_assert(std::is_standard_layout_v<T>);
				model.write((const char*)&data, sizeof(data));
			}
		};
		write_impl(write_impl, data, args...);
	};
	// write header (version number, type info, etc)
	write("HNSW", 4);
	write(uint32_t(3)); // version
	write(typeid(U).hash_code()^sizeof(U));
	write(sizeof(node));
	// write parameter configuration
	write(dim);
	write(m_l);
	write(m);
	write(ef_construction);
	write(alpha);
	write(n);
	// write indices
	for(const auto &u : node_pool)
	{
		write(u.level);
		write(U::get_id(u.data));
	}
	for(const auto &u : node_pool)
	{
		for(uint32_t l=0; l<=u.level; ++l)
		{
			write(u.neighbors[l].size());
			for(node_id pv : u.neighbors[l])
				write(pv);
		}
	}
	// write entrances
	write(entrance.size());
	for(node_id pu : entrance)
		write(pu);
} 
#endif

} // namespace HNSW

#endif // _HNSW_HPP

