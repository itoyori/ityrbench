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
  ityr::global_vector<long> bin_edges;

  ityr::global_vector<long> dest_id_bin_sizes;
  ityr::global_vector<long> update_bin_sizes;

  ityr::global_vector<ityr::global_vector<uintE>> dest_id_bins;
  ityr::global_vector<ityr::global_span<uintE>>   dest_id_bins_ref; // references to bins in other parts

  ityr::global_vector<ityr::global_vector<double>> update_bins;
  ityr::global_vector<ityr::global_span<double>>   update_bins_ref; // references to bins in other parts
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

  ityr::root_exec([=]() {
    ityr::execution::parallel_policy par_v {.cutoff_count = cutoff_v, .checkout_count = cutoff_v};
    ityr::execution::parallel_policy par_e {.cutoff_count = cutoff_e, .checkout_count = cutoff_e};

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

ityr::global_vector<part> partition(const graph& g) {
  auto t0 = ityr::gettime_ns();

  auto n_parts = g.n_parts;

  ityr::global_vector<part> parts(global_vec_coll_opts(1), n_parts);
  ityr::global_span<part> parts_ref(parts.data(), parts.size());

  auto n = g.n;
  ityr::global_span<vertex_data> v_out_data {g.v_out_data.data(), g.v_out_data.size()};
  ityr::global_span<uintE> out_edges {g.out_edges.data(), g.out_edges.size()};

  ityr::root_exec([=]() {
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
              ityr::execution::sequenced_policy{.checkout_count = cutoff_v},
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              ityr::make_global_iterator(&v_out_data[p.v_end]  , ityr::checkout_mode::read),
              [&](vertex_data v) {
                auto prev_bin = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy{.checkout_count = cutoff_e},
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_bin = (dest_vid >> bin_offset_bits);
                      assert(dest_bin < n_parts);

                      dest_id_bin_sizes[dest_bin]++;
                      if (dest_bin != prev_bin) {
                        update_bin_sizes[dest_bin]++;
                        prev_bin = dest_bin;
                      }
                    });
              });

          // prefix sum
          auto bin_edge_offsets = ityr::make_checkout(p.bin_edge_offsets.data(), p.bin_edge_offsets.size(), ityr::checkout_mode::write);

          bin_edge_offsets[0] = 0;
          for (auto j = 0; j < n_parts; j++) {
            bin_edge_offsets[j + 1] = bin_edge_offsets[j] + update_bin_sizes[j];
          }
          p.bin_edges.resize(bin_edge_offsets[n_parts]);

          auto bin_edges = ityr::make_checkout(p.bin_edges.data(), p.bin_edges.size(), ityr::checkout_mode::write);

          std::vector<long> bin_offsets(n_parts);

          ityr::for_each(
              ityr::execution::sequenced_policy{.checkout_count = cutoff_v},
              ityr::count_iterator<uintE>(p.v_begin),
              ityr::count_iterator<uintE>(p.v_end),
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              [&](uintE vid, vertex_data v) {
                auto prev_bin = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy{.checkout_count = cutoff_e},
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_bin = (dest_vid >> bin_offset_bits);
                      assert(dest_bin < n_parts);

                      if (dest_bin != prev_bin) {
                        bin_edges[bin_edge_offsets[dest_bin] + (bin_offsets[dest_bin]++)] = vid;
                        prev_bin = dest_bin;
                      }
                    });
              });

          p.dest_id_bins.resize(n_parts);
          p.dest_id_bins_ref.resize(n_parts);

          p.update_bins.resize(n_parts);
          p.update_bins_ref.resize(n_parts);
        });

    // store references to bins
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin(), ityr::checkout_mode::read),
        ityr::make_global_iterator(parts_ref.end()  , ityr::checkout_mode::read),
        [=](const part& p) {
          ityr::for_each(
              ityr::execution::sequenced_policy{.checkout_count = std::size_t(n_parts)},
              ityr::make_global_iterator(parts_ref.begin()         , ityr::checkout_mode::read),
              ityr::make_global_iterator(parts_ref.end()           , ityr::checkout_mode::read),
              ityr::make_global_iterator(p.dest_id_bins_ref.begin(), ityr::checkout_mode::write),
              ityr::make_global_iterator(p.update_bins_ref.begin() , ityr::checkout_mode::write),
              [&](const part& p2, ityr::global_span<uintE>& dest_id_bin_ref, ityr::global_span<double>& update_bin_ref) {
                auto dest_id_bin      = ityr::make_checkout(&p2.dest_id_bins[p.id]     , 1, ityr::checkout_mode::read_write);
                auto update_bin       = ityr::make_checkout(&p2.update_bins[p.id]      , 1, ityr::checkout_mode::read_write);
                auto dest_id_bin_size = ityr::make_checkout(&p2.dest_id_bin_sizes[p.id], 1, ityr::checkout_mode::read);
                auto update_bin_size  = ityr::make_checkout(&p2.update_bin_sizes[p.id] , 1, ityr::checkout_mode::read);

                dest_id_bin[0].resize(dest_id_bin_size[0]);
                update_bin[0].resize(update_bin_size[0]);

                dest_id_bin_ref = ityr::global_span<uintE>{dest_id_bin[0].data(), dest_id_bin[0].size()};
                update_bin_ref = ityr::global_span<double>{update_bin[0].data(), update_bin[0].size()};
              });
        });

    // write dest id
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin(), ityr::checkout_mode::read),
        ityr::make_global_iterator(parts_ref.end()  , ityr::checkout_mode::read),
        [=](const part& p) {
          auto dest_id_bins = ityr::make_checkout(p.dest_id_bins.data(), p.dest_id_bins.size(), ityr::checkout_mode::read);

          std::vector<long> offsets(n_parts);

          ityr::for_each(
              ityr::execution::sequenced_policy{.checkout_count = cutoff_v},
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              ityr::make_global_iterator(&v_out_data[p.v_end]  , ityr::checkout_mode::read),
              [&](vertex_data v) {
                auto prev_bin = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy{.checkout_count = cutoff_e},
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_bin = (dest_vid >> bin_offset_bits);
                      assert(dest_bin < n_parts);

                      if (dest_bin != prev_bin) {
                        dest_vid |= MAX_NEG;
                        prev_bin = dest_bin;
                      }

                      // TODO: course-grained checkout
                      dest_id_bins[dest_bin][offsets[dest_bin]++] = dest_vid;
                    });
              });
        });
  });

  auto t1 = ityr::gettime_ns();

  if (ityr::is_master()) {
    printf("Partitioning done (%ld ns).\n", t1 - t0);
    printf("n_parts = %ld\n", n_parts);
    printf("\n");
    fflush(stdout);
  }

  std::size_t total_bin_size = ityr::root_exec([=] {
    return ityr::transform_reduce(
        ityr::execution::par,
        parts_ref.begin(), parts_ref.end(),
        std::size_t(0), std::plus<>{},
        [=](const part& p) {
          return ityr::transform_reduce(
              ityr::execution::sequenced_policy{.checkout_count = std::size_t(n_parts)},
              p.dest_id_bins.begin(), p.dest_id_bins.end(),
              p.update_bins.begin(),
              std::size_t(0), std::plus<>{},
              [&](const ityr::global_vector<uintE>& dest_id_bin, const ityr::global_vector<double>& update_bin) {
                return dest_id_bin.size() * sizeof(uintE) + update_bin.size() * sizeof(double);
              });
        });
  });

  if (ityr::is_master()) {
    printf("Total bin size = %ld bytes\n", total_bin_size);
    printf("\n");
    fflush(stdout);
  }

  return parts;
}

void init_gpop(graph& g) {
  if (!is_pow2(bin_width)) {
    if (ityr::is_master()) {
      printf("bin_width (%ld) must be a power of 2.\n", bin_width);
    }
    exit(1);
  }

  // clear to save memory
  /* g.in_edges = {}; */
  /* g.v_in_data = {}; */

  g.n_parts = (g.n + bin_width - 1) / bin_width;
  g.parts = partition(g);
}

using neighbors = ityr::global_span<uintE>;

void pagerank_naive(const graph&              g,
                    ityr::global_span<double> p_curr,
                    ityr::global_span<double> p_next,
                    ityr::global_span<double> p_div,
                    ityr::global_span<double> p_div_next,
                    double                    eps = 0.000001) {
  const double damping = 0.85;
  auto n = g.n;
  const double addedConstant = (1 - damping) * (1 / static_cast<double>(n));

  double one_over_n = 1 / static_cast<double>(n);

  auto in_edges_begin = g.in_edges.begin();

  ityr::execution::parallel_policy par_v {.cutoff_count = cutoff_v, .checkout_count = cutoff_v};
  ityr::execution::parallel_policy par_e {.cutoff_count = cutoff_e, .checkout_count = cutoff_e};

  ityr::fill(par_v, p_curr.begin(), p_curr.end(), one_over_n);
  ityr::fill(par_v, p_next.begin(), p_next.end(), 0);

  ityr::transform(
      par_v,
      g.v_out_data.begin(), g.v_out_data.end(),
      p_div.begin(),
      [=](const vertex_data& vout) {
        return one_over_n / static_cast<double>(vout.degree);
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

          double contribution =
            ityr::transform_reduce(
                par_e,
                nghs.begin(), nghs.end(),
                double(0), std::plus<double>{},
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
        [=](const double& pn, const vertex_data& vout) {
          return pn / static_cast<double>(vout.degree);
        });

    double L1_norm =
      ityr::transform_reduce(
          par_v,
          p_curr.begin(), p_curr.end(),
          p_next.begin(),
          double(0), std::plus<double>{},
          [=](const double& pc, const double& pn) {
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

  double max_pr =
    ityr::reduce(
        par_v,
        p_next.begin(), p_next.end(),
        std::numeric_limits<double>::lowest(),
        [](double a, double b) { return std::max(a, b); });

  std::cout << "max_pr = " << max_pr << " iter = " << iter << std::endl;
}

void pagerank_gpop(graph&                    g,
                   ityr::global_span<double> p_curr,
                   ityr::global_span<double> p_next,
                   ityr::global_span<double> p_div,
                   ityr::global_span<double> p_div_next,
                   double                    eps = 0.000001) {
  const double damping = 0.85;
  auto n = g.n;
  const double added_constant = (1 - damping) * (1 / static_cast<double>(n));

  auto n_parts = g.n_parts;

  double one_over_n = 1 / static_cast<double>(n);

  ityr::execution::parallel_policy par_v {.cutoff_count = cutoff_v, .checkout_count = cutoff_v};

  ityr::fill(par_v, p_curr.begin(), p_curr.end(), one_over_n);
  ityr::fill(par_v, p_next.begin(), p_next.end(), 0);

  ityr::transform(
      par_v,
      g.v_out_data.begin(), g.v_out_data.end(),
      p_div.begin(),
      [=](const vertex_data& vout) {
        return one_over_n / static_cast<double>(vout.degree);
      });

  int iter = 0;
  while (iter++ < max_iters) {
    auto t0 = ityr::gettime_ns();

    // scatter
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(g.parts.begin(), ityr::checkout_mode::read),
        ityr::make_global_iterator(g.parts.end()  , ityr::checkout_mode::read),
        [=](const part& p) {
          auto p_div_ = ityr::make_checkout(&p_div[p.v_begin], p.n, ityr::checkout_mode::read);

          ityr::for_each(
              ityr::execution::sequenced_policy{.checkout_count = std::size_t(n_parts)},
              ityr::make_global_iterator(p.update_bins.begin()     , ityr::checkout_mode::read),
              ityr::make_global_iterator(p.update_bins.end()       , ityr::checkout_mode::read),
              ityr::make_global_iterator(p.bin_edge_offsets.begin(), ityr::checkout_mode::read),
              [&](const ityr::global_vector<double>& update_bins, long e_begin) {

                ityr::for_each(
                    ityr::execution::sequenced_policy{.checkout_count = cutoff_e},
                    ityr::make_global_iterator(update_bins.begin()  , ityr::checkout_mode::write),
                    ityr::make_global_iterator(update_bins.end()    , ityr::checkout_mode::write),
                    ityr::make_global_iterator(&p.bin_edges[e_begin], ityr::checkout_mode::read),
                    [&](double& update, uintE vid) {
                      assert(vid >= p.v_begin);
                      assert(vid - p.v_begin < p.n);
                      update = p_div_[vid - p.v_begin];
                    });
              });
        });

    auto t1 = ityr::gettime_ns();

    // gather
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(g.parts.begin(), ityr::checkout_mode::read),
        ityr::make_global_iterator(g.parts.end()  , ityr::checkout_mode::read),
        [=](const part& p) {
          auto p_next_ = ityr::make_checkout(&p_next[p.v_begin], p.n, ityr::checkout_mode::write);

          for (auto& pn : p_next_) {
            pn = 0;
          }

          ityr::for_each(
              ityr::execution::sequenced_policy{.checkout_count = std::size_t(n_parts)},
              ityr::make_global_iterator(p.dest_id_bins_ref.begin(), ityr::checkout_mode::read),
              ityr::make_global_iterator(p.dest_id_bins_ref.end()  , ityr::checkout_mode::read),
              ityr::make_global_iterator(p.update_bins_ref.begin() , ityr::checkout_mode::read),
              [&](const ityr::global_span<uintE>& dest_bin, const ityr::global_span<double>& update_bin) {

                if (update_bin.size() > 0) {
                  auto update_bin_ = ityr::make_checkout(update_bin.data(), update_bin.size(), ityr::checkout_mode::read);

                  long update_bin_offset = -1;
                  ityr::for_each(
                      ityr::execution::sequenced_policy{.checkout_count = dest_bin.size()},
                      ityr::make_global_iterator(dest_bin.begin(), ityr::checkout_mode::read),
                      ityr::make_global_iterator(dest_bin.end()  , ityr::checkout_mode::read),
                      [&](uintE dest_vid) {
                        update_bin_offset += (dest_vid >> MSB_ROT);
                        dest_vid &= MAX_POS;

                        assert(dest_vid >= p.v_begin);
                        assert(dest_vid - p.v_begin < p.n);
                        assert(std::size_t(update_bin_offset) < update_bin.size());
                        p_next_[dest_vid - p.v_begin] += update_bin_[update_bin_offset];
                      });
                }
              });

          for (auto& pn : p_next_) {
            pn = damping * pn + added_constant;
          }
        });

    auto t2 = ityr::gettime_ns();
    printf("scatter: %ld ns\n", t1 - t0);
    printf("gather:  %ld ns\n", t2 - t1);
    fflush(stdout);

    ityr::transform(
        par_v,
        p_next.begin(), p_next.end(),
        g.v_out_data.begin(),
        p_div_next.begin(),
        [=](const double& pn, const vertex_data& vout) {
          return pn / static_cast<double>(vout.degree);
        });

    double L1_norm =
      ityr::transform_reduce(
          par_v,
          p_curr.begin(), p_curr.end(),
          p_next.begin(),
          double(0), std::plus<double>{},
          [=](const double& pc, const double& pn) {
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

  double max_pr =
    ityr::reduce(
        par_v,
        p_next.begin(), p_next.end(),
        std::numeric_limits<double>::lowest(),
        [](double a, double b) { return std::max(a, b); });

  std::cout << "max_pr = " << max_pr << " iter = " << iter << std::endl;
}

void run() {
  static std::optional<graph> g = load_dataset(dataset_filename);
  if (exec_type == exec_t::Gpop) {
    init_gpop(*g);
  }

  ityr::global_vector<double> p_curr    (global_vec_coll_opts(cutoff_v), g->n);
  ityr::global_vector<double> p_next    (global_vec_coll_opts(cutoff_v), g->n);
  ityr::global_vector<double> p_div     (global_vec_coll_opts(cutoff_v), g->n);
  ityr::global_vector<double> p_div_next(global_vec_coll_opts(cutoff_v), g->n);

  for (int r = 0; r < n_repeats; r++) {
    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    ityr::root_exec([&]{
      if (exec_type == exec_t::Naive) {
        pagerank_naive(*g, ityr::global_span<double>{p_curr.data()    , p_curr.size()    },
                           ityr::global_span<double>{p_next.data()    , p_next.size()    },
                           ityr::global_span<double>{p_div.data()     , p_div.size()     },
                           ityr::global_span<double>{p_div_next.data(), p_div_next.size()});
      } else if (exec_type == exec_t::Gpop) {
        pagerank_gpop(*g, ityr::global_span<double>{p_curr.data()    , p_curr.size()    },
                          ityr::global_span<double>{p_next.data()    , p_next.size()    },
                          ityr::global_span<double>{p_div.data()     , p_div.size()     },
                          ityr::global_span<double>{p_div_next.data(), p_div_next.size()});
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
           "Max iterations:               %d\n"
           "Dataset:                      %s\n"
           "Cutoff for vertices:          %ld\n"
           "Cutoff for edges:             %ld\n"
           "Execution type:               %s\n"
           "Bin width (for gpop):         %ld\n"
           "-------------------------------------------------------------\n",
           ityr::n_ranks(), max_iters, dataset_filename, cutoff_v, cutoff_e,
           to_str(exec_type).c_str(), bin_width);

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
