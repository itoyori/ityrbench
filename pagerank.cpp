#include "common.hpp"

enum class exec_t {
  Naive = 0,
  Gpop = 1,
};

std::ostream& operator<<(std::ostream& o, const exec_t& e) {
  switch (e) {
    case exec_t::Naive: o << "naive"; break;
    case exec_t::Gpop: o << "gpop"; break;
  }
  return o;
}

template <typename T>
std::string to_str(T x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

/* #define MAX_NEG 0x8000000000000000 */
/* #define MAX_POS 0x7fffffffffffffff */
/* #define MSB_ROT 63 */
#define MAX_NEG 0x80000000
#define MAX_POS 0x7fffffff
#define MSB_ROT 31

using uintT = unsigned long;
using uintE = unsigned int;

#ifndef ITYRBENCH_REAL_TYPE
#define ITYRBENCH_REAL_TYPE double
#endif
using real_t = ITYRBENCH_REAL_TYPE;

struct vertex_data {
  std::size_t offset;
  uintE       degree;
};

struct part {
  long id;

  uintE v_begin;
  uintE v_end;

  long n;
  long m;

  ityr::global_vector<long> bin_edge_offsets;
  ityr::global_span<uintE> bin_edges;

  ityr::global_vector<long> dest_id_bin_sizes;
  ityr::global_vector<long> update_bin_sizes;

  ityr::global_vector<ityr::global_span<uintE>> dest_id_bins_read;
  ityr::global_vector<ityr::global_span<uintE>> dest_id_bins_write;

  ityr::global_vector<ityr::global_span<real_t>> update_bins_read;
  ityr::global_vector<ityr::global_span<real_t>> update_bins_write;

  std::size_t dest_id_bins_read_size;
};

struct graph {
  long n;
  long m;

  ityr::global_vector<vertex_data> v_in_data;
  ityr::global_vector<vertex_data> v_out_data;

  ityr::global_vector<uintE> in_edges;
  ityr::global_vector<uintE> out_edges;

  long n_parts;
  ityr::global_vector<part> parts;

  ityr::global_vector<uintE>  bin_edges;
  ityr::global_vector<uintE>  dest_id_bins;
  ityr::global_vector<real_t> update_bins;
};

int         n_repeats        = 10;
int         max_iters        = 100;
const char* dataset_filename = nullptr;
std::size_t cutoff_v         = 4096;
std::size_t cutoff_e         = 4096;
exec_t      exec_type        = exec_t::Naive;
long        bin_width        = 16 * 1024;
long        bin_offset_bits  = log2_pow2(bin_width);

ityr::global_vector_options global_vec_coll_opts(std::size_t cutoff_count) {
  return {
    .collective         = true,
    .parallel_construct = true,
    .parallel_destruct  = true,
    .cutoff_count       = cutoff_count,
  };
}

graph load_dataset(const char* filename) {
  auto t0 = ityr::gettime_ns();

  // TODO: these mmap things should be moved to Itoyori runtime code
  int fd = open(filename, O_RDONLY);
  if (fd == -1) {
    perror("open");
    abort();
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    abort();
  }

  std::size_t size = sb.st_size;

  void* p = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) {
    perror("mmap");
    abort();
  }

  std::byte* data = reinterpret_cast<std::byte*>(p);

  long n = reinterpret_cast<long*>(data)[0];
  long m = reinterpret_cast<long*>(data)[1];

  std::size_t skip = 3 * sizeof(long);

  // Put `static` keyword here so that the variable has the same virtual address
  // TODO: better handle this issue
  static uintT* out_offsets = reinterpret_cast<uintT*>(data + skip);
  skip += (n + 1) * sizeof(uintT);

  static uintE* out_edges = reinterpret_cast<uintE*>(data + skip);
  skip += m * sizeof(uintE);

  skip += 3 * sizeof(long);

  static uintT* in_offsets = reinterpret_cast<uintT*>(data + skip);
  skip += (n + 1) * sizeof(uintT);

  static uintE* in_edges = reinterpret_cast<uintE*>(data + skip);
  skip += m * sizeof(uintE);

  ityr::global_vector<vertex_data> v_in_data(global_vec_coll_opts(cutoff_v), n);
  ityr::global_vector<vertex_data> v_out_data(global_vec_coll_opts(cutoff_v), n);

  ityr::global_vector<uintE> in_edges_vec(global_vec_coll_opts(cutoff_e), m);
  ityr::global_vector<uintE> out_edges_vec(global_vec_coll_opts(cutoff_e), m);

  auto v_in_data_begin  = v_in_data.begin();
  auto v_out_data_begin = v_out_data.begin();

  auto in_edges_vec_begin  = in_edges_vec.begin();
  auto out_edges_vec_begin = out_edges_vec.begin();

  ityr::root_exec([=] {
    ityr::execution::parallel_policy par_v(cutoff_v);
    ityr::execution::parallel_policy par_e(cutoff_e);

    ityr::transform(
        par_v,
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(n),
        v_in_data_begin,
        [&](long i) {
          return vertex_data {
            .offset = in_offsets[i],
            .degree = static_cast<uintE>(in_offsets[i + 1] - in_offsets[i]),
          };
        });

    ityr::transform(
        par_v,
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(n),
        v_out_data_begin,
        [&](long i) {
          return vertex_data {
            .offset = out_offsets[i],
            .degree = static_cast<uintE>(out_offsets[i + 1] - out_offsets[i]),
          };
        });

    ityr::transform(
        par_e,
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(m),
        in_edges_vec_begin,
        [&](long i) { return in_edges[i]; });

    ityr::transform(
        par_e,
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(m),
        out_edges_vec_begin,
        [&](long i) { return out_edges[i]; });
  });

  if (close(fd) == -1) {
    perror("close");
    abort();
  }

  if (munmap(p, size) == -1) {
    perror("munmap");
    abort();
  }

  ityr::barrier();

  auto t1 = ityr::gettime_ns();

  if (ityr::is_master()) {
    printf("Dataset loaded (%ld ns).\n", t1 - t0);
    printf("N = %ld M = %ld\n", n, m);
    printf("\n");
    fflush(stdout);
  }

  return {
    .n          = n,
    .m          = m,
    .v_in_data  = std::move(v_in_data),
    .v_out_data = std::move(v_out_data),
    .in_edges   = std::move(in_edges_vec),
    .out_edges  = std::move(out_edges_vec),
    .n_parts    = 0,
    .parts      = {},
  };
}

void partition(long n_parts, graph& g) {
  auto t0 = ityr::gettime_ns();

  g.n_parts = n_parts;

  ityr::global_vector<part> parts(global_vec_coll_opts(1), n_parts);
  ityr::global_span<part> parts_ref(parts);

  auto n = g.n;
  ityr::global_span<vertex_data> v_out_data(g.v_out_data);
  ityr::global_span<uintE> out_edges(g.out_edges);

  ityr::root_exec([=] {
    ityr::for_each(
        ityr::execution::par,
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(n_parts),
        ityr::make_global_iterator(parts_ref.begin(), ityr::checkout_mode::read_write),
        [=](long pid, part& p) {
          p.id      = pid;
          p.v_begin = pid * bin_width;
          p.v_end   = std::min(n, (pid + 1) * bin_width);
          p.n       = p.v_end - p.v_begin;

          vertex_data vb = v_out_data[p.v_begin];
          vertex_data ve = v_out_data[p.v_end - 1];
          p.m = ve.degree + ve.offset - vb.offset;

          p.bin_edge_offsets.resize(n_parts + 1);

          p.dest_id_bin_sizes.resize(n_parts);
          p.update_bin_sizes.resize(n_parts);

          auto dest_id_bin_sizes = ityr::make_checkout(p.dest_id_bin_sizes.data(), p.dest_id_bin_sizes.size(), ityr::checkout_mode::read_write);
          auto update_bin_sizes  = ityr::make_checkout(p.update_bin_sizes.data() , p.update_bin_sizes.size() , ityr::checkout_mode::read_write);

          ityr::for_each(
              ityr::execution::sequenced_policy(cutoff_v),
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              ityr::make_global_iterator(&v_out_data[p.v_end]  , ityr::checkout_mode::read),
              [&](vertex_data v) {
                auto prev_bin = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy(cutoff_e),
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_id_bin = (dest_vid >> bin_offset_bits);
                      assert(dest_id_bin < n_parts);

                      dest_id_bin_sizes[dest_id_bin]++;
                      if (dest_id_bin != prev_bin) {
                        update_bin_sizes[dest_id_bin]++;
                        prev_bin = dest_id_bin;
                      }
                    });
              });

          // prefix sum
          auto bin_edge_offsets = ityr::make_checkout(p.bin_edge_offsets.data(), p.bin_edge_offsets.size(), ityr::checkout_mode::write);

          bin_edge_offsets[0] = 0;
          for (auto j = 0; j < n_parts; j++) {
            bin_edge_offsets[j + 1] = bin_edge_offsets[j] + update_bin_sizes[j];
          }
        });
  });

  /* Allocate bin edges */
  ityr::global_vector<std::size_t> bin_edges_offsets(global_vec_coll_opts(1), n_parts + 1);
  ityr::global_span<std::size_t> bin_edges_offsets_ref(bin_edges_offsets);

  ityr::root_exec([=] {
    bin_edges_offsets_ref.front().put(0);

    ityr::transform_inclusive_scan(
        ityr::execution::par,
        parts_ref.begin(), parts_ref.end(),
        bin_edges_offsets_ref.begin() + 1,
        ityr::reducer::plus<std::size_t>{},
        [=](const part& p) {
          return p.bin_edge_offsets[n_parts].get();
        });
  });

  std::size_t bin_edges_size_total = bin_edges_offsets.back().get();

  if (ityr::is_master()) {
    printf("bin edges size   = %ld bytes\n", bin_edges_size_total * sizeof(uintE));
    fflush(stdout);
  }

  ityr::global_vector<uintE> bin_edges(global_vec_coll_opts(cutoff_e), bin_edges_size_total);
  ityr::global_span<uintE> bin_edges_ref(bin_edges);

  ityr::root_exec([=] {
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin()            , ityr::checkout_mode::read_write),
        ityr::make_global_iterator(parts_ref.end()              , ityr::checkout_mode::read_write),
        ityr::make_global_iterator(bin_edges_offsets_ref.begin(), ityr::checkout_mode::read),
        [=](part& p, std::size_t bin_edges_offset) {
          auto bin_edge_offsets = ityr::make_checkout(p.bin_edge_offsets.data(), p.bin_edge_offsets.size(), ityr::checkout_mode::read);

          std::size_t bin_edges_size = bin_edge_offsets[n_parts];
          p.bin_edges = bin_edges_ref.subspan(bin_edges_offset, bin_edges_size);

          auto bin_edges = ityr::make_checkout(p.bin_edges.data(), p.bin_edges.size(), ityr::checkout_mode::write);

          std::vector<long> bin_offsets(n_parts);

          ityr::for_each(
              ityr::execution::sequenced_policy(cutoff_v),
              ityr::count_iterator<uintE>(p.v_begin),
              ityr::count_iterator<uintE>(p.v_end),
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              [&](uintE vid, vertex_data v) {
                auto prev_bin = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy(cutoff_e),
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_id_bin = (dest_vid >> bin_offset_bits);
                      assert(dest_id_bin < n_parts);

                      if (dest_id_bin != prev_bin) {
                        bin_edges[bin_edge_offsets[dest_id_bin] + (bin_offsets[dest_id_bin]++)] = vid;
                        prev_bin = dest_id_bin;
                      }
                    });
              });

          p.dest_id_bins_read.resize(n_parts);
          p.dest_id_bins_write.resize(n_parts);

          p.update_bins_read.resize(n_parts);
          p.update_bins_write.resize(n_parts);
        });
  });

  /* Allocate dest_id bins */
  ityr::global_vector<std::size_t> dest_id_bin_offsets(global_vec_coll_opts(1), n_parts + 1);
  ityr::global_span<std::size_t> dest_id_bin_offsets_ref(dest_id_bin_offsets);

  ityr::root_exec([=] {
    dest_id_bin_offsets_ref.front().put(0);

    ityr::transform_inclusive_scan(
        ityr::execution::par,
        ityr::count_iterator<long>(0), ityr::count_iterator<long>(n_parts),
        dest_id_bin_offsets_ref.begin() + 1,
        ityr::reducer::plus<std::size_t>{},
        [=](long pid) {
          return ityr::transform_reduce(
              ityr::execution::sequenced_policy(n_parts),
              parts_ref.begin(), parts_ref.end(), ityr::reducer::plus<std::size_t>{},
              [=](const part& p) {
                return p.dest_id_bin_sizes[pid].get();
              });
        });
  });

  std::size_t dest_id_bin_size_total = dest_id_bin_offsets.back().get();

  if (ityr::is_master()) {
    printf("dest_id bin size = %ld bytes\n", dest_id_bin_size_total * sizeof(uintE));
    fflush(stdout);
  }

  ityr::global_vector<uintE> dest_id_bins(global_vec_coll_opts(cutoff_e), dest_id_bin_size_total);
  ityr::global_span<uintE> dest_id_bins_ref(dest_id_bins);

  /* Allocate update bins */
  ityr::global_vector<std::size_t> update_bin_offsets(global_vec_coll_opts(1), n_parts + 1);
  ityr::global_span<std::size_t> update_bin_offsets_ref(update_bin_offsets);

  ityr::root_exec([=] {
    update_bin_offsets_ref.front().put(0);

    ityr::transform_inclusive_scan(
        ityr::execution::par,
        ityr::count_iterator<long>(0), ityr::count_iterator<long>(n_parts),
        update_bin_offsets_ref.begin() + 1,
        ityr::reducer::plus<std::size_t>{},
        [=](long pid) {
          return ityr::transform_reduce(
              ityr::execution::sequenced_policy(n_parts),
              parts_ref.begin(), parts_ref.end(), ityr::reducer::plus<std::size_t>{},
              [=](const part& p) {
                return p.update_bin_sizes[pid].get();
              });
        });
  });

  std::size_t update_bin_size_total = update_bin_offsets.back().get();

  if (ityr::is_master()) {
    printf("update bin size  = %ld bytes\n", update_bin_size_total * sizeof(real_t));
    fflush(stdout);
  }

  ityr::global_vector<real_t> update_bins(global_vec_coll_opts(cutoff_e), update_bin_size_total);
  ityr::global_span<real_t> update_bins_ref(update_bins);

  /* Distribute bins to parts */
  ityr::root_exec([=] {
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin()              , ityr::checkout_mode::read),
        ityr::make_global_iterator(parts_ref.end()                , ityr::checkout_mode::read),
        ityr::make_global_iterator(dest_id_bin_offsets_ref.begin(), ityr::checkout_mode::read),
        ityr::make_global_iterator(update_bin_offsets_ref.begin() , ityr::checkout_mode::read),
        [=](part& p, std::size_t dest_id_bin_offset, std::size_t update_bin_offset) {

          std::size_t d_offset = dest_id_bin_offset;
          std::size_t u_offset = update_bin_offset;

          // Set spans for bins to read from
          ityr::for_each(
              ityr::execution::sequenced_policy(n_parts),
              ityr::make_global_iterator(parts_ref.begin()          , ityr::checkout_mode::read),
              ityr::make_global_iterator(parts_ref.end()            , ityr::checkout_mode::read),
              ityr::make_global_iterator(p.dest_id_bins_read.begin(), ityr::checkout_mode::write),
              ityr::make_global_iterator(p.update_bins_read.begin() , ityr::checkout_mode::write),
              [&](const part& p2, ityr::global_span<uintE>& dest_id_bin, ityr::global_span<real_t>& update_bin) {
                std::size_t dest_id_bin_size = p2.dest_id_bin_sizes[p.id].get();
                dest_id_bin = dest_id_bins_ref.subspan(d_offset, dest_id_bin_size);
                d_offset += dest_id_bin_size;

                std::size_t update_bin_size = p2.update_bin_sizes[p.id].get();
                update_bin = update_bins_ref.subspan(u_offset, update_bin_size);
                u_offset += update_bin_size;
              });
        });

    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin(), ityr::checkout_mode::read),
        ityr::make_global_iterator(parts_ref.end()  , ityr::checkout_mode::read),
        [=](part& p) {
          // Set spans for bins to write to
          ityr::for_each(
              ityr::execution::sequenced_policy(n_parts),
              ityr::make_global_iterator(parts_ref.begin()           , ityr::checkout_mode::read),
              ityr::make_global_iterator(parts_ref.end()             , ityr::checkout_mode::read),
              ityr::make_global_iterator(p.dest_id_bins_write.begin(), ityr::checkout_mode::write),
              ityr::make_global_iterator(p.update_bins_write.begin() , ityr::checkout_mode::write),
              [&](const part& p2, ityr::global_span<uintE>& dest_id_bin, ityr::global_span<real_t>& update_bin) {
                // bin_write[i][j] = bin_read[j][i]
                dest_id_bin = p2.dest_id_bins_read[p.id].get();
                update_bin = p2.update_bins_read[p.id].get();
              });
        });

    // write dest id
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin(), ityr::checkout_mode::read),
        ityr::make_global_iterator(parts_ref.end()  , ityr::checkout_mode::read),
        [=](part& p) {
          auto dest_id_bins = ityr::make_checkout(p.dest_id_bins_write.data(), p.dest_id_bins_write.size(),
                                                  ityr::checkout_mode::read);

          std::vector<long> offsets(n_parts);

          ityr::for_each(
              ityr::execution::sequenced_policy(cutoff_v),
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              ityr::make_global_iterator(&v_out_data[p.v_end]  , ityr::checkout_mode::read),
              [&](vertex_data v) {
                auto prev_bin = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy(cutoff_e),
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_id_bin = (dest_vid >> bin_offset_bits);
                      assert(dest_id_bin < n_parts);

                      if (dest_id_bin != prev_bin) {
                        dest_vid |= MAX_NEG;
                        prev_bin = dest_id_bin;
                      }

                      // TODO: course-grained checkout
                      dest_id_bins[dest_id_bin][offsets[dest_id_bin]++].put(dest_vid);
                    });
              });
        });

    // calculate the total dest_id bin size (for reading) for each part
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin(), ityr::checkout_mode::read_write),
        ityr::make_global_iterator(parts_ref.end()  , ityr::checkout_mode::read_write),
        [=](part& p) {
          p.dest_id_bins_read_size = ityr::transform_reduce(
              ityr::execution::sequenced_policy(n_parts),
              p.dest_id_bins_read.begin(), p.dest_id_bins_read.end(),
              ityr::reducer::plus<std::size_t>{},
              [](const ityr::global_span<uintE>& s) { return s.size(); });
        });
  });

  auto t1 = ityr::gettime_ns();

  if (ityr::is_master()) {
    printf("Partitioning done (%ld ns).\n", t1 - t0);
    printf("n_parts = %ld\n", n_parts);
    printf("\n");
    fflush(stdout);
  }

  g.parts        = std::move(parts);
  g.bin_edges    = std::move(bin_edges);
  g.dest_id_bins = std::move(dest_id_bins);
  g.update_bins  = std::move(update_bins);
}

void init_gpop(graph& g) {
  if (!is_pow2(bin_width)) {
    if (ityr::is_master()) {
      printf("bin_width (%ld) must be a power of 2.\n", bin_width);
    }
    exit(1);
  }

  // clear to save memory
  g.in_edges = {};
  g.v_in_data = {};

  long n_parts = (g.n + bin_width - 1) / bin_width;
  partition(n_parts, g);
}

using neighbors = ityr::global_span<uintE>;

void pagerank_naive(graph&                    g,
                    ityr::global_span<real_t> p_curr,
                    ityr::global_span<real_t> p_next,
                    ityr::global_span<real_t> p_div,
                    ityr::global_span<real_t> p_div_next,
                    real_t                    eps = 0.000001) {
  const real_t damping = 0.85;
  auto n = g.n;
  const real_t addedConstant = (1 - damping) * (1 / static_cast<real_t>(n));

  real_t one_over_n = 1 / static_cast<real_t>(n);

  auto in_edges_begin = g.in_edges.begin();

  ityr::execution::parallel_policy par_v(cutoff_v);
  ityr::execution::parallel_policy par_e(cutoff_e);

  ityr::fill(par_v, p_curr.begin(), p_curr.end(), one_over_n);
  ityr::fill(par_v, p_next.begin(), p_next.end(), 0);

  ityr::transform(
      par_v,
      g.v_out_data.begin(), g.v_out_data.end(),
      p_div.begin(),
      [=](const vertex_data& vout) {
        return one_over_n / static_cast<real_t>(vout.degree);
      });

  int iter = 0;
  while (iter++ < max_iters) {
    ityr::transform(
        ityr::execution::par,
        ityr::make_global_iterator(g.v_in_data.begin(), ityr::checkout_mode::no_access),
        ityr::make_global_iterator(g.v_in_data.end()  , ityr::checkout_mode::no_access),
        ityr::make_global_iterator(p_next.begin()     , ityr::checkout_mode::no_access),
        [=](auto vin_) {
          vertex_data vin = vin_;

          neighbors nghs = neighbors{in_edges_begin + vin.offset, vin.degree};

          real_t contribution = ityr::transform_reduce(
              par_e,
              nghs.begin(), nghs.end(),
              ityr::reducer::plus<real_t>{},
              [=](const uintE& idx) {
                return p_div[idx].get();
              });

          return damping * contribution + addedConstant;
        });

    ityr::transform(
        par_v,
        p_next.begin(), p_next.end(),
        g.v_out_data.begin(),
        p_div_next.begin(),
        [=](const real_t& pn, const vertex_data& vout) {
          return pn / static_cast<real_t>(vout.degree);
        });

    real_t L1_norm = ityr::transform_reduce(
        par_v,
        p_curr.begin(), p_curr.end(),
        p_next.begin(),
        ityr::reducer::plus<real_t>{},
        [=](const real_t& pc, const real_t& pn) {
          return fabs(pc - pn);
        });

    if (L1_norm < eps) break;

    /* std::cout << "L1_norm = " << L1_norm << std::endl; */

    std::swap(p_curr, p_next);
    std::swap(p_div, p_div_next);
  }

  if (iter > max_iters) {
    std::swap(p_curr, p_next);
    iter--;
  }

  real_t max_pr = ityr::reduce(
      par_v,
      p_next.begin(), p_next.end(),
      ityr::reducer::max<real_t>{});

  std::cout << "max_pr = " << max_pr << " iter = " << iter << std::endl;
}

void pagerank_gpop(graph&                    g,
                   ityr::global_span<real_t> p_curr,
                   ityr::global_span<real_t> p_next,
                   ityr::global_span<real_t> p_div,
                   ityr::global_span<real_t> p_div_next,
                   real_t                    eps = 0.000001) {
  const real_t damping = 0.85;
  auto n = g.n;
  auto m = g.m;
  const real_t added_constant = (1 - damping) * (1 / static_cast<real_t>(n));

  auto n_parts = g.n_parts;

  real_t one_over_n = 1 / static_cast<real_t>(n);

  ityr::execution::parallel_policy par_v(cutoff_v);

  ityr::fill(par_v, p_curr.begin(), p_curr.end(), one_over_n);
  ityr::fill(par_v, p_next.begin(), p_next.end(), 0);

  ityr::transform(
      par_v,
      g.v_out_data.begin(), g.v_out_data.end(),
      p_div.begin(),
      [=](const vertex_data& vout) {
        return one_over_n / static_cast<real_t>(vout.degree);
      });

  auto workhint_scatter = ityr::create_workhint_range(
      ityr::execution::par,
      g.parts.begin(), g.parts.end(),
      [=](const part& p) {
        return p.bin_edges.size();
      });

  auto workhint_gather = ityr::create_workhint_range(
      ityr::execution::par,
      g.parts.begin(), g.parts.end(),
      [=](const part& p) {
        return p.dest_id_bins_read_size;
      });

  int iter = 0;
  while (iter++ < max_iters) {
    auto t0 = ityr::gettime_ns();

    // scatter
    ityr::for_each(
        ityr::execution::parallel_policy(workhint_scatter),
        g.parts.begin(), g.parts.end(),
        [=](auto&& p_ref) {
          auto p_cs = ityr::make_checkout(&p_ref, 1, ityr::checkout_mode::read);
          auto pn                = p_cs[0].n;
          auto v_begin           = p_cs[0].v_begin;
          auto bin_edge_offsets  = ityr::global_span<long>(p_cs[0].bin_edge_offsets);
          auto bin_edges         = p_cs[0].bin_edges;
          auto update_bins_write = ityr::global_span<ityr::global_span<real_t>>(p_cs[0].update_bins_write);
          p_cs.checkin();

          ityr::for_each(
              ityr::execution::par,
              ityr::make_global_iterator(update_bins_write.begin(), ityr::checkout_mode::read),
              ityr::make_global_iterator(update_bins_write.end()  , ityr::checkout_mode::read),
              ityr::make_global_iterator(bin_edge_offsets.begin() , ityr::checkout_mode::read),
              [=](ityr::global_span<real_t>& update_bin, long e_begin) {

                auto p_div_ = ityr::make_checkout(&p_div[v_begin], pn, ityr::checkout_mode::read);

                ityr::for_each(
                    ityr::execution::sequenced_policy(cutoff_e),
                    ityr::make_global_iterator(update_bin.begin(), ityr::checkout_mode::write),
                    ityr::make_global_iterator(update_bin.end()  , ityr::checkout_mode::write),
                    ityr::make_global_iterator(&bin_edges[e_begin], ityr::checkout_mode::read),
                    [&](real_t& update, uintE vid) {
                      assert(vid >= v_begin);
                      assert(vid - v_begin < pn);
                      update = p_div_[vid - v_begin];
                    });
              });
        });

    auto t1 = ityr::gettime_ns();

    // gather
    ityr::for_each(
        ityr::execution::parallel_policy(workhint_gather),
        g.parts.begin(), g.parts.end(),
        [=](auto&& p_ref) {
          auto p_cs = ityr::make_checkout(&p_ref, 1, ityr::checkout_mode::read);
          auto pn                = p_cs[0].n;
          auto v_begin           = p_cs[0].v_begin;
          auto dest_id_bins_read = ityr::global_span<ityr::global_span<uintE>>(p_cs[0].dest_id_bins_read);
          auto update_bins_read  = ityr::global_span<ityr::global_span<real_t>>(p_cs[0].update_bins_read);
          bool do_parallel       = p_cs[0].dest_id_bins_read_size > std::size_t(m / n_parts);
          p_cs.checkin();

          if (do_parallel) {
            auto gather_reducer = ityr::reducer::make_reducer(
                [=]() {
                  return ityr::global_vector<real_t>(pn, 0);
                },
                [=](ityr::global_vector<real_t>& acc, std::pair<ityr::global_span<uintE>, ityr::global_span<real_t>> sp) {
                  auto dest_id_bin = sp.first;
                  auto update_bin  = sp.second;

                  auto acc_ = ityr::make_checkout(acc.data(), acc.size(), ityr::checkout_mode::read_write);

                  if (update_bin.size() > 0) {
                    auto update_bin_ = ityr::make_checkout(update_bin.data(), update_bin.size(), ityr::checkout_mode::read);

                    long update_bin_offset = -1;
                    ityr::for_each(
                        ityr::execution::sequenced_policy(dest_id_bin.size()),
                        ityr::make_global_iterator(dest_id_bin.begin(), ityr::checkout_mode::read),
                        ityr::make_global_iterator(dest_id_bin.end()  , ityr::checkout_mode::read),
                        [&](uintE dest_vid) {
                          update_bin_offset += (dest_vid >> MSB_ROT);
                          dest_vid &= MAX_POS;

                          assert(dest_vid >= v_begin);
                          assert(dest_vid - v_begin < pn);
                          assert(std::size_t(update_bin_offset) < update_bin.size());
                          acc_[dest_vid - v_begin] += update_bin_[update_bin_offset];
                        });
                  }
                },
                [=](ityr::global_vector<real_t>& acc1, const ityr::global_vector<real_t>& acc2) {
                  ityr::transform(
                      ityr::execution::sequenced_policy(acc1.size()),
                      acc1.begin(), acc1.end(), acc2.begin(), acc1.begin(),
                      std::plus<>{});
                });

            auto p_sum = ityr::transform_reduce(
                ityr::execution::par,
                dest_id_bins_read.begin(), dest_id_bins_read.end(),
                update_bins_read.begin(),
                gather_reducer,
                [=](const ityr::global_span<uintE>& dest_bin, const ityr::global_span<real_t>& update_bin) {
                  return std::make_pair(dest_bin, update_bin);
                });

            ityr::transform(
                ityr::execution::sequenced_policy(pn),
                p_sum.begin(), p_sum.end(), p_next.begin() + v_begin,
                [=](real_t x) { return damping * x + added_constant; });

          } else {
            auto p_next_ = ityr::make_checkout(&p_next[v_begin], pn, ityr::checkout_mode::write);

            for (auto& x : p_next_) {
              x = 0;
            }

            ityr::for_each(
                ityr::execution::sequenced_policy(n_parts),
                ityr::make_global_iterator(dest_id_bins_read.begin(), ityr::checkout_mode::read),
                ityr::make_global_iterator(dest_id_bins_read.end()  , ityr::checkout_mode::read),
                ityr::make_global_iterator(update_bins_read.begin() , ityr::checkout_mode::read),
                [&](const ityr::global_span<uintE>& dest_id_bin, const ityr::global_span<real_t>& update_bin) {

                  if (update_bin.size() > 0) {
                    auto update_bin_ = ityr::make_checkout(update_bin.data(), update_bin.size(), ityr::checkout_mode::read);

                    long update_bin_offset = -1;
                    ityr::for_each(
                        ityr::execution::sequenced_policy(dest_id_bin.size()),
                        ityr::make_global_iterator(dest_id_bin.begin(), ityr::checkout_mode::read),
                        ityr::make_global_iterator(dest_id_bin.end()  , ityr::checkout_mode::read),
                        [&](uintE dest_vid) {
                          update_bin_offset += (dest_vid >> MSB_ROT);
                          dest_vid &= MAX_POS;

                          assert(dest_vid >= v_begin);
                          assert(dest_vid - v_begin < pn);
                          assert(std::size_t(update_bin_offset) < update_bin.size());
                          p_next_[dest_vid - v_begin] += update_bin_[update_bin_offset];
                        });
                  }
                });

            for (auto& x : p_next_) {
              x = damping * x + added_constant;
            }
          }
        });

    auto t2 = ityr::gettime_ns();
    printf("scatter: %ld ns\n", t1 - t0);
    printf("gather:  %ld ns\n", t2 - t1);

    ityr::transform(
        par_v,
        p_next.begin(), p_next.end(),
        g.v_out_data.begin(),
        p_div_next.begin(),
        [=](const real_t& pn, const vertex_data& vout) {
          return pn / static_cast<real_t>(vout.degree);
        });

    real_t L1_norm = ityr::transform_reduce(
        par_v,
        p_curr.begin(), p_curr.end(),
        p_next.begin(),
        ityr::reducer::plus<real_t>{},
        [=](const real_t& pc, const real_t& pn) {
          return fabs(pc - pn);
        });

    if (L1_norm < eps) break;

    /* std::cout << "L1_norm = " << L1_norm << std::endl; */

    std::swap(p_curr, p_next);
    std::swap(p_div, p_div_next);
  }

  if (iter > max_iters) {
    std::swap(p_curr, p_next);
    iter--;
  }

  real_t max_pr = ityr::reduce(
      par_v,
      p_next.begin(), p_next.end(),
      ityr::reducer::max<real_t>{});

  std::cout << "max_pr = " << max_pr << " iter = " << iter << std::endl;
}

void run() {
  static std::optional<graph> g = load_dataset(dataset_filename);
  if (exec_type == exec_t::Gpop) {
    init_gpop(*g);
  }

  ityr::global_vector<real_t> p_curr_vec    (global_vec_coll_opts(cutoff_v), g->n);
  ityr::global_vector<real_t> p_next_vec    (global_vec_coll_opts(cutoff_v), g->n);
  ityr::global_vector<real_t> p_div_vec     (global_vec_coll_opts(cutoff_v), g->n);
  ityr::global_vector<real_t> p_div_next_vec(global_vec_coll_opts(cutoff_v), g->n);

  ityr::global_span<real_t> p_curr    (p_curr_vec);
  ityr::global_span<real_t> p_next    (p_next_vec);
  ityr::global_span<real_t> p_div     (p_div_vec);
  ityr::global_span<real_t> p_div_next(p_div_next_vec);

  for (int r = 0; r < n_repeats; r++) {
    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    ityr::root_exec([&] {
      if (exec_type == exec_t::Naive) {
        pagerank_naive(*g, p_curr, p_next, p_div, p_div_next);
      } else if (exec_type == exec_t::Gpop) {
        pagerank_gpop(*g, p_curr, p_next, p_div, p_div_next);
      }
    });

    auto t1 = ityr::gettime_ns();

    ityr::profiler_end();

    if (ityr::is_master()) {
      printf("[%d] %ld ns\n", r, t1 - t0);
      fflush(stdout);
    }

    ityr::profiler_flush();
  }

  g = std::nullopt; // call destructor here
}

void show_help_and_exit(int argc [[maybe_unused]], char** argv) {
  if (ityr::is_master()) {
    printf("Usage: %s [options]\n"
           "  options:\n"
           "    -r : # of repeats (int)\n"
           "    -i : # of maximum iterations (int)\n"
           "    -f : path to the dataset binary file (string)\n"
           "    -v : cutoff count for vertices (size_t)\n"
           "    -e : cutoff count for edges (size_t)\n"
           "    -t : execution type (0: naive, 1: gpop)\n"
           "    -b : bin width (power of 2) for gpop (long)\n", argv[0]);
  }
  exit(1);
}

int main(int argc, char** argv) {
  ityr::init();

  set_signal_handlers();

  int opt;
  while ((opt = getopt(argc, argv, "r:i:f:v:e:t:b:h")) != EOF) {
    switch (opt) {
      case 'r':
        n_repeats = atoi(optarg);
        break;
      case 'i':
        max_iters = atoi(optarg);
        break;
      case 'f':
        dataset_filename = optarg;
        break;
      case 'v':
        cutoff_v = atol(optarg);
        break;
      case 'e':
        cutoff_e = atol(optarg);
        break;
      case 't':
        exec_type = exec_t(atoi(optarg));
        break;
      case 'b':
        bin_width = atol(optarg);
        bin_offset_bits = log2_pow2(bin_width);
        break;
      default:
        show_help_and_exit(argc, argv);
    }
  }

  if (!dataset_filename) {
    if (ityr::is_master()) {
      printf("Please specify the dataset (-i).\n");
    }
    exit(1);
  }

  if (ityr::is_master()) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[PageRank]\n"
           "# of processes:               %d\n"
           "Real number type:             %s (%ld bytes)\n"
           "Max iterations:               %d\n"
           "Dataset:                      %s\n"
           "Cutoff for vertices:          %ld\n"
           "Cutoff for edges:             %ld\n"
           "Execution type:               %s\n"
           "Bin width (for gpop):         %ld\n"
           "-------------------------------------------------------------\n",
           ityr::n_ranks(), typename_str<real_t>(), sizeof(real_t), max_iters, dataset_filename,
           cutoff_v, cutoff_e, to_str(exec_type).c_str(), bin_width);

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

  run();

  ityr::fini();
  return 0;
}
