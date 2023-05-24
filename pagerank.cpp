#include "common.hpp"

using uintT = unsigned long;
using uintE = unsigned int;

struct vertex_data {
  std::size_t offset;
  uintE       degree;
};

struct graph {
  long n;
  long m;

  ityr::global_vector<vertex_data> v_in_data;
  ityr::global_vector<vertex_data> v_out_data;

  ityr::global_vector<uintE> in_edges;
  ityr::global_vector<uintE> out_edges;
};

int         n_repeats        = 10;
const char* dataset_filename = nullptr;

ityr::global_vector_options global_vec_coll_opts {
  .collective         = true,
  .parallel_construct = true,
  .parallel_destruct  = true,
  .cutoff_count       = 4096,
};

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

  uintT* out_offsets = reinterpret_cast<uintT*>(data + skip);
  skip += (n + 1) * sizeof(uintT);

  uintE* out_edges = reinterpret_cast<uintE*>(data + skip);
  skip += m * sizeof(uintE);

  skip += 3 * sizeof(long);

  uintT* in_offsets = reinterpret_cast<uintT*>(data + skip);
  skip += (n + 1) * sizeof(uintT);

  uintE* in_edges = reinterpret_cast<uintE*>(data + skip);
  skip += m * sizeof(uintE);

  ityr::global_vector<vertex_data> v_out_data(global_vec_coll_opts, n + 1);
  ityr::global_vector<vertex_data> v_in_data(global_vec_coll_opts, n + 1);

  // TODO: really size of `m`?
  ityr::global_vector<uintE> out_edges_vec(global_vec_coll_opts, m);
  ityr::global_vector<uintE> in_edges_vec(global_vec_coll_opts, m);

  ityr::root_exec([&]() {
    ityr::parallel_for_each(
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(n),
        ityr::make_global_iterator(v_out_data.begin(), ityr::ori::mode::write),
        [&](long i, vertex_data& vd) {
      vd.offset = out_offsets[i];
      vd.degree = out_offsets[i + 1] - out_offsets[i];
    });

    ityr::parallel_for_each(
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(n),
        ityr::make_global_iterator(v_in_data.begin(), ityr::ori::mode::write),
        [&](long i, vertex_data& vd) {
      vd.offset = in_offsets[i];
      vd.degree = in_offsets[i + 1] - in_offsets[i];
    });

    ityr::parallel_for_each(
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(m),
        ityr::make_global_iterator(out_edges_vec.begin(), ityr::ori::mode::write),
        [&](long i, uintE& e) {
      e = out_edges[i];
    });

    ityr::parallel_for_each(
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(m),
        ityr::make_global_iterator(in_edges_vec.begin(), ityr::ori::mode::write),
        [&](long i, uintE& e) {
      e = in_edges[i];
    });
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
    fflush(stdout);
  }

  return {
    .n          = n,
    .m          = m,
    .v_in_data  = std::move(v_in_data),
    .v_out_data = std::move(v_out_data),
    .in_edges   = std::move(in_edges_vec),
    .out_edges  = std::move(out_edges_vec),
  };
}

using neighbors = ityr::global_span<uintE>;

struct pr_vertex {
  double p[2];
  double p_div[2];

  neighbors in_nghs;
  neighbors out_nghs;

  neighbors in_neighbors() const { return in_nghs; }
  neighbors out_neighbors() const { return out_nghs; }

  uintE in_degree() { return in_nghs.size(); }
  uintE out_degree() { return out_nghs.size(); }
};

void pagerank(const graph&                 g,
              ityr::global_span<pr_vertex> pr_vertices,
              double                       eps       = 0.000001,
              std::size_t                  max_iters = 100) {
  const double damping = 0.85;
  auto n = g.n;
  const double addedConstant = (1 - damping) * (1 / static_cast<double>(n));

  double one_over_n = 1 / static_cast<double>(n);

  auto in_edges_begin  = g.in_edges.begin();
  auto out_edges_begin = g.out_edges.begin();

  ityr::parallel_for_each(
      {.cutoff_count = 1024, .checkout_count = 1024},
      ityr::make_global_iterator(pr_vertices.begin() , ityr::ori::mode::write),
      ityr::make_global_iterator(pr_vertices.end()   , ityr::ori::mode::write),
      ityr::make_global_iterator(g.v_in_data.begin() , ityr::ori::mode::read),
      ityr::make_global_iterator(g.v_out_data.begin(), ityr::ori::mode::read),
      [=](pr_vertex& pv, const vertex_data& vin, const vertex_data& vout) {
        pv.in_nghs  = {in_edges_begin + vin.offset, vin.degree};
        pv.out_nghs = {out_edges_begin + vout.offset, vout.degree};
        pv.p[0]     = one_over_n;
        pv.p[1]     = 0;
        pv.p_div[0] = one_over_n / static_cast<double>(vout.degree);
        pv.p_div[1] = 0;
      });

  int flip = 0;

  size_t iter = 0;
  while (iter++ < max_iters) {
    ityr::parallel_for_each(
        {.cutoff_count = 1, .checkout_count = 1},
        ityr::make_global_iterator(pr_vertices.begin(), ityr::ori::mode::no_access),
        ityr::make_global_iterator(pr_vertices.end()  , ityr::ori::mode::no_access),
        [=](ityr::ori::global_ref<pr_vertex> ur) {
          pr_vertex u = ur;

          auto nghs = u.in_neighbors();

          double contribution =
            ityr::parallel_reduce(
                {.cutoff_count = 1024, .checkout_count = 1024},
                nghs.begin(), nghs.end(),
                double(0), std::plus<double>{},
                [=](const uintE& i) {
                  pr_vertex v = pr_vertices[i];
                  return v.p_div[flip];
                });

          u.p[!flip] = damping * contribution + addedConstant;
          u.p_div[!flip] = u.p[!flip] / static_cast<double>(u.out_degree());

          ur = u;
        });

    double L1_norm =
      ityr::parallel_reduce(
          {.cutoff_count = 1024, .checkout_count = 1024},
          pr_vertices.begin(), pr_vertices.end(),
          double(0), std::plus<double>{},
          [=](const pr_vertex& v) {
            return fabs(v.p[flip] - v.p[!flip]);
          });

    if (L1_norm < eps) break;

    /* std::cout << "L1_norm = " << L1_norm << std::endl; */

    flip = !flip;
  }

  double max_pr =
    ityr::parallel_reduce(
        {.cutoff_count = 1024, .checkout_count = 1024},
        pr_vertices.begin(), pr_vertices.end(),
        std::numeric_limits<double>::lowest(),
        [](double a, double b) { return std::max(a, b); },
        [=](const pr_vertex& v) {
          return v.p[!flip];
        });

  std::cout << "max_pr = " << max_pr << " iter = " << iter << std::endl;
}

void run() {
  graph g = load_dataset(dataset_filename);
  ityr::global_vector<pr_vertex> pr_vertices(global_vec_coll_opts, g.n);

  for (int r = 0; r < n_repeats; r++) {
    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    ityr::root_exec([&]{
      pagerank(g, {pr_vertices.data(), pr_vertices.size()});
    });

    auto t1 = ityr::gettime_ns();

    ityr::profiler_end();

    if (ityr::is_master()) {
      printf("[%d] %ld ns\n", r, t1 - t0);
      fflush(stdout);
    }

    ityr::profiler_flush();
  }
}

void show_help_and_exit(int argc [[maybe_unused]], char** argv) {
  if (ityr::is_master()) {
    printf("Usage: %s [options]\n"
           "  options:\n"
           "    -r : # of repeats (int)\n"
           "    -i : path to the dataset binary file (string)\n", argv[0]);
  }
  exit(1);
}

int main(int argc, char** argv) {
  ityr::init();

  set_signal_handlers();

  int opt;
  while ((opt = getopt(argc, argv, "r:i:h")) != EOF) {
    switch (opt) {
      case 'r':
        n_repeats = atoi(optarg);
        break;
      case 'i':
        dataset_filename = optarg;
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
           "Dataset:                      %s\n"
           "-------------------------------------------------------------\n",
           ityr::n_ranks(), dataset_filename);

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
