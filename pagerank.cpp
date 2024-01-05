#include "common.hpp"

struct prof_event_user_scatter : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_scatter"; }
};

struct prof_event_user_gather_l1 : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_gather_l1"; }
};

struct prof_event_user_gather_l2 : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_gather_l2"; }
};

struct prof_event_user_gather_l3 : public ityr::common::profiler::event {
  using event::event;
  std::string str() const override { return "user_gather_l3"; }
};

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

using bin_edge_elem_t = uint16_t;

using dest_bin_elem_t = uint32_t;

inline constexpr long max_bin_width = 1 << (sizeof(dest_bin_elem_t) * 8 / 2);

using dest_bin_normal = ityr::global_span<dest_bin_elem_t>;

struct dest_sub_bin {
  dest_bin_normal               elems;
  std::pair<uint32_t, uint32_t> offset_range;
};

using dest_bin_subs = ityr::global_vector<dest_sub_bin>;

using dest_bin = std::variant<dest_bin_normal, dest_bin_subs>;

using update_bin = ityr::global_span<real_t>;

struct part {
  long id;

  uintE v_begin;
  uintE v_end;

  long n;
  long m;

  ityr::global_vector<long> bin_edge_offsets;
  ityr::global_span<bin_edge_elem_t> bin_edges;

  ityr::global_vector<long> dest_bin_sizes;
  ityr::global_vector<long> update_bin_sizes;

  ityr::global_vector<dest_bin> dest_bins_read;
  ityr::global_vector<dest_bin> dest_bins_write;

  ityr::global_vector<update_bin> update_bins_read;
  ityr::global_vector<update_bin> update_bins_write;

  ityr::global_vector<ityr::global_vector<dest_bin_normal>> sub_dest_bins;

  std::size_t dest_bins_read_size;
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

  ityr::global_vector<bin_edge_elem_t> bin_edges;
  ityr::global_vector<dest_bin_elem_t> dest_bin_elems;
  ityr::global_vector<real_t>          update_bin_elems;

  ityr::workhint_range<std::size_t> workhint_scatter;
  ityr::workhint_range<std::size_t> workhint_gather;
};

int         n_repeats        = 10;
int         max_iters        = 100;
const char* dataset_filename = nullptr;
std::size_t cutoff_v         = 4096;
std::size_t cutoff_e         = std::size_t(16) * 1024;
std::size_t cutoff_g         = std::size_t(1024) * 1024;
exec_t      exec_type        = exec_t::Naive;
long        bin_width        = 16 * 1024;
long        bin_offset_bits  = log2_pow2(bin_width);

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

  ityr::global_vector<vertex_data> v_in_data(ityr::global_vector_options(true, cutoff_v));
  ityr::global_vector<uintE> in_edges_vec(ityr::global_vector_options(true, cutoff_e));

  if (exec_type == exec_t::Naive) {
    v_in_data.resize(n);
    in_edges_vec.resize(m);
  }

  ityr::global_vector<vertex_data> v_out_data(ityr::global_vector_options(true, cutoff_v), n);
  ityr::global_vector<uintE> out_edges_vec(ityr::global_vector_options(true, cutoff_e), m);

  auto v_in_data_begin = v_in_data.begin();
  auto in_edges_vec_begin = in_edges_vec.begin();

  auto v_out_data_begin = v_out_data.begin();
  auto out_edges_vec_begin = out_edges_vec.begin();

  ityr::root_exec([=] {
    ityr::execution::parallel_policy par_v(cutoff_v);
    ityr::execution::parallel_policy par_e(cutoff_e);

    if (exec_type == exec_t::Naive) {
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
          par_e,
          ityr::count_iterator<long>(0),
          ityr::count_iterator<long>(m),
          in_edges_vec_begin,
          [&](long i) { return in_edges[i]; });
    }

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

  graph g;
  g.n          = n;
  g.m          = m;
  g.v_in_data  = std::move(v_in_data);
  g.v_out_data = std::move(v_out_data);
  g.in_edges   = std::move(in_edges_vec);
  g.out_edges  = std::move(out_edges_vec);
  g.n_parts    = 0;
  return g;
}

dest_bin_subs partition_sub_bins(ityr::ori::global_ptr<dest_bin_elem_t> first,
                                 ityr::ori::global_ptr<dest_bin_elem_t> last,
                                 uint32_t                               offset_b,
                                 uint32_t                               offset_e) {
  if (std::size_t(last - first) < cutoff_g) {
    return dest_bin_subs(1, {dest_bin_normal(first, last),
                             std::make_pair(offset_b, offset_e)});
  }

  uint32_t offset_m = offset_b + (offset_e - offset_b) / 2;
  auto mid = ityr::stable_partition(
      ityr::execution::parallel_policy(cutoff_g),
      first, last,
      [=](dest_bin_elem_t dest) {
        auto dest_offset = dest & 0xffff;
        return dest_offset < offset_m;
      });

  auto [db1, db2] = ityr::parallel_invoke(
      [=] { return partition_sub_bins(first, mid , offset_b, offset_m); },
      [=] { return partition_sub_bins(mid  , last, offset_m, offset_e); });

  db1.insert(db1.end(), db2.begin(), db2.end());
  return db1;
}

void partition(long n_parts, graph& g) {
  auto t0 = ityr::gettime_ns();

  g.n_parts = n_parts;

  ityr::global_vector<part> parts(ityr::global_vector_options(true, 1), n_parts);
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

          p.dest_bin_sizes.resize(n_parts);
          p.update_bin_sizes.resize(n_parts);

          auto dest_bin_sizes   = ityr::make_checkout(p.dest_bin_sizes.data()  , p.dest_bin_sizes.size()  , ityr::checkout_mode::read_write);
          auto update_bin_sizes = ityr::make_checkout(p.update_bin_sizes.data(), p.update_bin_sizes.size(), ityr::checkout_mode::read_write);

          ityr::for_each(
              ityr::execution::sequenced_policy(cutoff_v),
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              ityr::make_global_iterator(&v_out_data[p.v_end]  , ityr::checkout_mode::read),
              [&](vertex_data v) {
                auto prev_bin_id = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy(cutoff_e),
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_bin_id = (dest_vid >> bin_offset_bits);
                      assert(dest_bin_id < n_parts);

                      dest_bin_sizes[dest_bin_id]++;
                      if (dest_bin_id != prev_bin_id) {
                        update_bin_sizes[dest_bin_id]++;
                        prev_bin_id = dest_bin_id;
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
  ityr::global_vector<std::size_t> bin_edges_offsets(ityr::global_vector_options(true, 1), n_parts + 1);
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
    printf("bin edges size  = %ld bytes\n", bin_edges_size_total * sizeof(bin_edge_elem_t));
    fflush(stdout);
  }

  ityr::global_vector<bin_edge_elem_t> bin_edges(ityr::global_vector_options(true, cutoff_e), bin_edges_size_total);
  ityr::global_span<bin_edge_elem_t> bin_edges_ref(bin_edges);

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

          auto bin_edges_ = ityr::make_checkout(p.bin_edges.data(), p.bin_edges.size(), ityr::checkout_mode::write);

          std::vector<long> bin_offsets(n_parts);

          ityr::for_each(
              ityr::execution::sequenced_policy(cutoff_v),
              ityr::count_iterator<uintE>(p.v_begin),
              ityr::count_iterator<uintE>(p.v_end),
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              [&](uintE vid, vertex_data v) {
                auto prev_bin_id = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy(cutoff_e),
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_bin_id = (dest_vid >> bin_offset_bits);
                      assert(dest_bin_id < n_parts);

                      if (dest_bin_id != prev_bin_id) {
                        bin_edges_[bin_edge_offsets[dest_bin_id] + (bin_offsets[dest_bin_id]++)] = vid - p.v_begin;
                        prev_bin_id = dest_bin_id;
                      }
                    });
              });

          p.dest_bins_read.resize(n_parts);
          p.dest_bins_write.resize(n_parts);

          p.update_bins_read.resize(n_parts);
          p.update_bins_write.resize(n_parts);
        });
  });

  /* Allocate dest bins */
  ityr::global_vector<std::size_t> dest_bin_offsets(ityr::global_vector_options(true, 1), n_parts + 1);
  ityr::global_span<std::size_t> dest_bin_offsets_ref(dest_bin_offsets);

  ityr::root_exec([=] {
    dest_bin_offsets_ref.front().put(0);

    ityr::transform_inclusive_scan(
        ityr::execution::par,
        ityr::count_iterator<long>(0), ityr::count_iterator<long>(n_parts),
        dest_bin_offsets_ref.begin() + 1,
        ityr::reducer::plus<std::size_t>{},
        [=](long pid) {
          return ityr::transform_reduce(
              ityr::execution::sequenced_policy(n_parts),
              parts_ref.begin(), parts_ref.end(), ityr::reducer::plus<std::size_t>{},
              [=](const part& p) {
                return p.dest_bin_sizes[pid].get();
              });
        });
  });

  std::size_t dest_bin_size_total = dest_bin_offsets.back().get();

  if (ityr::is_master()) {
    printf("dest bin size   = %ld bytes\n", dest_bin_size_total * sizeof(dest_bin_elem_t));
    fflush(stdout);
  }

  ityr::global_vector<dest_bin_elem_t> dest_bin_elems(ityr::global_vector_options(true, cutoff_e), dest_bin_size_total);
  ityr::global_span<dest_bin_elem_t> dest_bin_elems_ref(dest_bin_elems);

  /* Allocate update bins */
  ityr::global_vector<std::size_t> update_bin_offsets(ityr::global_vector_options(true, 1), n_parts + 1);
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
    printf("update bin size = %ld bytes\n", update_bin_size_total * sizeof(real_t));
    fflush(stdout);
  }

  ityr::global_vector<real_t> update_bin_elems(ityr::global_vector_options(true, cutoff_e), update_bin_size_total);
  ityr::global_span<real_t> update_bin_elems_ref(update_bin_elems);

  /* Distribute bins to parts */
  ityr::root_exec([=] {
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin()             , ityr::checkout_mode::read_write),
        ityr::make_global_iterator(parts_ref.end()               , ityr::checkout_mode::read_write),
        ityr::make_global_iterator(dest_bin_offsets_ref.begin()  , ityr::checkout_mode::read),
        ityr::make_global_iterator(update_bin_offsets_ref.begin(), ityr::checkout_mode::read),
        [=](part& p, std::size_t dest_bin_offset, std::size_t update_bin_offset) {

          std::size_t d_offset = dest_bin_offset;
          std::size_t u_offset = update_bin_offset;

          // Set spans for bins to read from
          ityr::for_each(
              ityr::execution::sequenced_policy(n_parts),
              ityr::make_global_iterator(parts_ref.begin()         , ityr::checkout_mode::read),
              ityr::make_global_iterator(parts_ref.end()           , ityr::checkout_mode::read),
              ityr::make_global_iterator(p.dest_bins_read.begin()  , ityr::checkout_mode::read_write),
              ityr::make_global_iterator(p.update_bins_read.begin(), ityr::checkout_mode::write),
              [&](const part& p2, dest_bin& d_bin, update_bin& u_bin) {
                std::size_t dest_bin_size = p2.dest_bin_sizes[p.id].get();
                auto s = dest_bin_elems_ref.subspan(d_offset, dest_bin_size);
                d_bin.emplace<dest_bin_normal>(s);
                d_offset += dest_bin_size;

                std::size_t update_bin_size = p2.update_bin_sizes[p.id].get();
                u_bin = update_bin_elems_ref.subspan(u_offset, update_bin_size);
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
              ityr::make_global_iterator(parts_ref.begin()          , ityr::checkout_mode::read),
              ityr::make_global_iterator(parts_ref.end()            , ityr::checkout_mode::read),
              ityr::make_global_iterator(p.dest_bins_write.begin()  , ityr::checkout_mode::read_write),
              ityr::make_global_iterator(p.update_bins_write.begin(), ityr::checkout_mode::write),
              [&](const part& p2, dest_bin& d_bin, update_bin& u_bin) {
                // bin_write[i][j] = bin_read[j][i]
                auto d_bin_cs = ityr::make_checkout(&p2.dest_bins_read[p.id], 1, ityr::checkout_mode::read);
                d_bin = d_bin_cs[0];
                u_bin = p2.update_bins_read[p.id].get();
              });
        });

    // write dest id
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin(), ityr::checkout_mode::read),
        ityr::make_global_iterator(parts_ref.end()  , ityr::checkout_mode::read),
        [=](part& p) {
          auto dest_bins = ityr::make_checkout(p.dest_bins_write.data(), p.dest_bins_write.size(),
                                               ityr::checkout_mode::read);

          std::vector<long> d_bin_offsets(n_parts);
          std::vector<dest_bin_elem_t> update_elem_offsets(n_parts, -1);

          ityr::for_each(
              ityr::execution::sequenced_policy(cutoff_v),
              ityr::make_global_iterator(&v_out_data[p.v_begin], ityr::checkout_mode::read),
              ityr::make_global_iterator(&v_out_data[p.v_end]  , ityr::checkout_mode::read),
              [&](vertex_data v) {
                auto prev_bin_id = n_parts + 1;

                auto e_begin = v.offset;
                auto e_end   = v.offset + v.degree;

                ityr::for_each(
                    ityr::execution::sequenced_policy(cutoff_e),
                    ityr::make_global_iterator(&out_edges[e_begin], ityr::checkout_mode::read),
                    ityr::make_global_iterator(&out_edges[e_end]  , ityr::checkout_mode::read),
                    [&](uintE dest_vid) {
                      uintE dest_bin_id = (dest_vid >> bin_offset_bits);
                      assert(dest_bin_id < n_parts);

                      dest_bin_elem_t dest_offset = dest_vid % bin_width;

                      if (dest_bin_id != prev_bin_id) {
                        update_elem_offsets[dest_bin_id]++;
                        prev_bin_id = dest_bin_id;
                      }

                      dest_bin_elem_t dest = (update_elem_offsets[dest_bin_id] << 16) + dest_offset;

                      // TODO: course-grained checkout
                      auto&& d_bin = std::get<dest_bin_normal>(dest_bins[dest_bin_id]);
                      d_bin[d_bin_offsets[dest_bin_id]++].put(dest);
                    });
              });
        });

    // store the total dest bin size (for reading) for each part
    ityr::for_each(
        ityr::execution::par,
        ityr::make_global_iterator(parts_ref.begin(), ityr::checkout_mode::read_write),
        ityr::make_global_iterator(parts_ref.end()  , ityr::checkout_mode::read_write),
        [=](part& p) {
          p.dest_bins_read_size = ityr::transform_reduce(
              ityr::execution::sequenced_policy(n_parts),
              p.dest_bins_read.begin(), p.dest_bins_read.end(),
              ityr::reducer::plus<std::size_t>{},
              [=](const dest_bin& d_bin) {
                return std::get<dest_bin_normal>(d_bin).size();
              });
        });

    // further partition large bins into sub-bins if needed
    ityr::for_each(
        ityr::execution::par,
        parts_ref.begin(), parts_ref.end(),
        [=](auto&& p_ref) {
          auto p_cs = ityr::make_checkout(&p_ref, 1, ityr::checkout_mode::read);
          auto pn             = p_cs[0].n;
          auto dest_bins_read = ityr::global_span<dest_bin>(p_cs[0].dest_bins_read);
          p_cs.checkin();

          ityr::for_each(
              ityr::execution::par,
              dest_bins_read.begin(), dest_bins_read.end(),
              [=](auto&& d_bin_ref) {
                auto d_bin_cs = ityr::make_checkout(&d_bin_ref, 1, ityr::checkout_mode::read);
                dest_bin_normal d_bin_orig = std::get<dest_bin_normal>(d_bin_cs[0]);
                d_bin_cs.checkin();

                if (d_bin_orig.size() > cutoff_g) {
                  // partition sub bins
                  dest_bin_subs sub_bins = partition_sub_bins(d_bin_orig.begin(), d_bin_orig.end(), 0, pn);

                  auto d_bin_cs = ityr::make_checkout(&d_bin_ref, 1, ityr::checkout_mode::read_write);
                  dest_bin& d_bin = d_bin_cs[0];
                  d_bin.emplace<dest_bin_subs>(std::move(sub_bins));
                }
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

  g.parts            = std::move(parts);
  g.bin_edges        = std::move(bin_edges);
  g.dest_bin_elems   = std::move(dest_bin_elems);
  g.update_bin_elems = std::move(update_bin_elems);
}

void init_gpop(graph& g) {
  if (!is_pow2(bin_width)) {
    if (ityr::is_master()) {
      printf("bin_width (%ld) must be a power of 2.\n", bin_width);
    }
    exit(1);
  }

  if (bin_width > max_bin_width) {
    if (ityr::is_master()) {
      printf("bin_width (%ld) cannot be larger than %ld.\n", bin_width, max_bin_width);
    }
    exit(1);
  }

  long n_parts = (g.n + bin_width - 1) / bin_width;
  partition(n_parts, g);

  g.workhint_scatter = ityr::create_workhint_range(
      ityr::execution::par,
      g.parts.begin(), g.parts.end(),
      [=](const part& p) {
        return p.bin_edges.size();
      });

  g.workhint_gather = ityr::create_workhint_range(
      ityr::execution::par,
      g.parts.begin(), g.parts.end(),
      [=](const part& p) {
        return p.dest_bins_read_size;
      });
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

  int iter = 0;
  while (iter++ < max_iters) {
    auto t0 = ityr::gettime_ns();

    // scatter
    ityr::for_each(
        ityr::execution::parallel_policy(g.workhint_scatter),
        g.parts.begin(), g.parts.end(),
        [=](auto&& p_ref) {
          auto p_cs = ityr::make_checkout(&p_ref, 1, ityr::checkout_mode::read);
          auto pn                = p_cs[0].n;
          auto v_begin           = p_cs[0].v_begin;
          auto bin_edge_offsets  = ityr::global_span<long>(p_cs[0].bin_edge_offsets);
          auto bin_edges         = p_cs[0].bin_edges;
          auto update_bins_write = ityr::global_span<update_bin>(p_cs[0].update_bins_write);
          p_cs.checkin();

          ityr::for_each(
              ityr::execution::par,
              ityr::make_global_iterator(update_bins_write.begin(), ityr::checkout_mode::read),
              ityr::make_global_iterator(update_bins_write.end()  , ityr::checkout_mode::read),
              ityr::make_global_iterator(bin_edge_offsets.begin() , ityr::checkout_mode::read),
              [=](update_bin& u_bin, long e_begin) {
                ITYR_PROFILER_RECORD(prof_event_user_scatter);

                auto p_div_ = ityr::make_checkout(&p_div[v_begin], pn, ityr::checkout_mode::read);

                if (u_bin.size() == 0) return;

                ityr::for_each(
                    ityr::execution::sequenced_policy(u_bin.size()),
                    ityr::make_global_iterator(u_bin.begin()      , ityr::checkout_mode::write),
                    ityr::make_global_iterator(u_bin.end()        , ityr::checkout_mode::write),
                    ityr::make_global_iterator(&bin_edges[e_begin], ityr::checkout_mode::read),
                    [&](real_t& update, bin_edge_elem_t vid_offset) {
                      assert(vid_offset < pn);
                      update = p_div_[vid_offset];
                    });
              });
        });

    auto t1 = ityr::gettime_ns();

    // gather
    ityr::for_each(
        ityr::execution::parallel_policy(g.workhint_gather),
        g.parts.begin(), g.parts.end(),
        [=](auto&& p_ref) {
          auto p_cs = ityr::make_checkout(&p_ref, 1, ityr::checkout_mode::read);
          auto pn               = p_cs[0].n;
          auto v_begin          = p_cs[0].v_begin;
          auto dest_bins_read   = ityr::global_span<dest_bin>(p_cs[0].dest_bins_read);
          auto update_bins_read = ityr::global_span<update_bin>(p_cs[0].update_bins_read);
          bool do_parallel      = p_cs[0].dest_bins_read_size > cutoff_g;
          p_cs.checkin();

          if (do_parallel) {
            auto gather_reducer = ityr::reducer::make_reducer(
                [=]() {
                  return ityr::global_vector<real_t>(pn, 0);
                },
                [=](ityr::global_vector<real_t>&                                                  acc,
                    std::pair<ityr::ori::global_ref<dest_bin>, ityr::ori::global_ref<update_bin>> ref_pair) {

                  auto u_bin = ref_pair.second.get();

                  if (u_bin.size() == 0) return;

                  auto d_bin_cs = ityr::make_checkout(&ref_pair.first, 1, ityr::checkout_mode::read);

                  if (std::holds_alternative<dest_bin_subs>(d_bin_cs[0])) {
                    auto&& d_bin = std::get<dest_bin_subs>(d_bin_cs[0]);

                    auto sub_bins_begin = d_bin.begin();
                    auto sub_bins_end   = d_bin.end();

                    d_bin_cs.checkin();

                    ityr::global_span<real_t> acc_ref(acc);

                    ityr::for_each(
                        ityr::execution::par,
                        ityr::make_global_iterator(sub_bins_begin, ityr::checkout_mode::read),
                        ityr::make_global_iterator(sub_bins_end  , ityr::checkout_mode::read),
                        [=](const dest_sub_bin& d_sub_bin) {
                          ITYR_PROFILER_RECORD(prof_event_user_gather_l3);

                          auto [offset_b, offset_e] = d_sub_bin.offset_range;

                          assert(offset_e <= pn);
                          assert(offset_b < offset_e);
                          std::size_t acc_offset = offset_b;
                          std::size_t acc_size = offset_e - offset_b;

                          auto u_bin_ = ityr::make_checkout(u_bin, ityr::checkout_mode::read);
                          auto acc_ = ityr::make_checkout(acc_ref.subspan(acc_offset, acc_size), ityr::checkout_mode::read_write);

                          ityr::for_each(
                              ityr::execution::sequenced_policy(d_sub_bin.elems.size()),
                              ityr::make_global_iterator(d_sub_bin.elems.begin(), ityr::checkout_mode::read),
                              ityr::make_global_iterator(d_sub_bin.elems.end()  , ityr::checkout_mode::read),
                              [&](dest_bin_elem_t dest) {
                                auto u_bin_offset = dest >> 16;
                                auto dest_offset  = dest & 0xffff;

                                assert(acc_offset <= dest_offset);
                                assert(dest_offset < acc_offset + acc_size);
                                assert(u_bin_offset < u_bin.size());
                                acc_[dest_offset - acc_offset] += u_bin_[u_bin_offset];
                              });
                        });

                  } else {
                    ITYR_PROFILER_RECORD(prof_event_user_gather_l2);

                    auto&& d_bin_ = std::get<dest_bin_normal>(d_bin_cs[0]);
                    auto u_bin_ = ityr::make_checkout(u_bin, ityr::checkout_mode::read);
                    auto acc_ = ityr::make_checkout(acc.data(), acc.size(), ityr::checkout_mode::read_write);

                    ityr::for_each(
                        ityr::execution::sequenced_policy(d_bin_.size()),
                        ityr::make_global_iterator(d_bin_.begin(), ityr::checkout_mode::read),
                        ityr::make_global_iterator(d_bin_.end()  , ityr::checkout_mode::read),
                        [&](dest_bin_elem_t dest) {
                          auto u_bin_offset = dest >> 16;
                          auto dest_offset  = dest & 0xffff;

                          assert(dest_offset < pn);
                          assert(u_bin_offset < u_bin.size());
                          acc_[dest_offset] += u_bin_[u_bin_offset];
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
                ityr::make_global_iterator(dest_bins_read.begin()  , ityr::checkout_mode::no_access),
                ityr::make_global_iterator(dest_bins_read.end()    , ityr::checkout_mode::no_access),
                ityr::make_global_iterator(update_bins_read.begin(), ityr::checkout_mode::no_access),
                gather_reducer,
                [=](auto&& d_bin_ref, auto&& u_bin_ref) {
                  return std::make_pair(d_bin_ref, u_bin_ref);
                });

            ityr::transform(
                ityr::execution::sequenced_policy(pn),
                p_sum.begin(), p_sum.end(), p_next.begin() + v_begin,
                [=](real_t x) { return damping * x + added_constant; });

          } else {
            ITYR_PROFILER_RECORD(prof_event_user_gather_l1);

            auto p_next_ = ityr::make_checkout(&p_next[v_begin], pn, ityr::checkout_mode::write);

            for (auto& x : p_next_) {
              x = 0;
            }

            ityr::for_each(
                ityr::execution::sequenced_policy(n_parts),
                ityr::make_global_iterator(dest_bins_read.begin()  , ityr::checkout_mode::read),
                ityr::make_global_iterator(dest_bins_read.end()    , ityr::checkout_mode::read),
                ityr::make_global_iterator(update_bins_read.begin(), ityr::checkout_mode::read),
                [&](const dest_bin& d_bin, const update_bin& u_bin) {
                  auto&& d_bin_ = std::get<dest_bin_normal>(d_bin);

                  if (u_bin.size() == 0) return;

                  auto u_bin_ = ityr::make_checkout(u_bin, ityr::checkout_mode::read);

                  ityr::for_each(
                      ityr::execution::sequenced_policy(d_bin_.size()),
                      ityr::make_global_iterator(d_bin_.begin(), ityr::checkout_mode::read),
                      ityr::make_global_iterator(d_bin_.end()  , ityr::checkout_mode::read),
                      [&](dest_bin_elem_t dest) {
                        auto u_bin_offset = dest >> 16;
                        auto dest_offset  = dest & 0xffff;

                        assert(dest_offset < pn);
                        assert(u_bin_offset < u_bin.size());
                        p_next_[dest_offset] += u_bin_[u_bin_offset];
                      });
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

  ityr::global_vector<real_t> p_curr_vec    (ityr::global_vector_options(true, cutoff_v), g->n);
  ityr::global_vector<real_t> p_next_vec    (ityr::global_vector_options(true, cutoff_v), g->n);
  ityr::global_vector<real_t> p_div_vec     (ityr::global_vector_options(true, cutoff_v), g->n);
  ityr::global_vector<real_t> p_div_next_vec(ityr::global_vector_options(true, cutoff_v), g->n);

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
           "    -g : cutoff count for leaf tasks in the gather phase (size_t)\n"
           "    -t : execution type (0: naive, 1: gpop)\n"
           "    -b : bin width (power of 2) for gpop (long)\n", argv[0]);
  }
  exit(1);
}

int main(int argc, char** argv) {
  ityr::init();

  ityr::common::profiler::event_initializer<prof_event_user_scatter>   ITYR_ANON_VAR;
  ityr::common::profiler::event_initializer<prof_event_user_gather_l1> ITYR_ANON_VAR;
  ityr::common::profiler::event_initializer<prof_event_user_gather_l2> ITYR_ANON_VAR;
  ityr::common::profiler::event_initializer<prof_event_user_gather_l3> ITYR_ANON_VAR;

  set_signal_handlers();

  int opt;
  while ((opt = getopt(argc, argv, "r:i:f:v:e:g:t:b:h")) != EOF) {
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
      case 'g':
        cutoff_g = atol(optarg);
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
           "Cutoff for gather leaf tasks: %ld\n"
           "Execution type:               %s\n"
           "Bin width (for gpop):         %ld\n"
           "-------------------------------------------------------------\n",
           ityr::n_ranks(), typename_str<real_t>(), sizeof(real_t), max_iters, dataset_filename,
           cutoff_v, cutoff_e, cutoff_g, to_str(exec_type).c_str(), bin_width);

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
