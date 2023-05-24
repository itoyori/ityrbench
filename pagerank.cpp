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

  ityr::global_vector<vertex_data> v_out_data(global_vec_coll_opts, n + 1);
  ityr::global_vector<vertex_data> v_in_data(global_vec_coll_opts, n + 1);

  // TODO: really size of `m`?
  ityr::global_vector<uintE> out_edges_vec(global_vec_coll_opts, m);
  ityr::global_vector<uintE> in_edges_vec(global_vec_coll_opts, m);

  ityr::root_exec([&]() {
    ityr::parallel_for_each(
        {.cutoff_count = 1024, .checkout_count = 1024},
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(n),
        ityr::make_global_iterator(v_out_data.begin(), ityr::ori::mode::write),
        [&](long i, vertex_data& vd) {
      vd.offset = out_offsets[i];
      vd.degree = out_offsets[i + 1] - out_offsets[i];
    });

    ityr::parallel_for_each(
        {.cutoff_count = 1024, .checkout_count = 1024},
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(n),
        ityr::make_global_iterator(v_in_data.begin(), ityr::ori::mode::write),
        [&](long i, vertex_data& vd) {
      vd.offset = in_offsets[i];
      vd.degree = in_offsets[i + 1] - in_offsets[i];
    });

    ityr::parallel_for_each(
        {.cutoff_count = 1024, .checkout_count = 1024},
        ityr::count_iterator<long>(0),
        ityr::count_iterator<long>(m),
        ityr::make_global_iterator(out_edges_vec.begin(), ityr::ori::mode::write),
        [&](long i, uintE& e) {
      e = out_edges[i];
    });

    ityr::parallel_for_each(
        {.cutoff_count = 1024, .checkout_count = 1024},
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
  };
}

using neighbors = ityr::global_span<uintE>;

void pagerank(const graph&              g,
              ityr::global_span<double> p_curr,
              ityr::global_span<double> p_next,
              ityr::global_span<double> p_div,
              ityr::global_span<double> p_div_next,
              ityr::global_span<double> differences,
              double                    eps       = 0.000001,
              std::size_t               max_iters = 100) {
  const double damping = 0.85;
  auto n = g.n;
  const double addedConstant = (1 - damping) * (1 / static_cast<double>(n));

  double one_over_n = 1 / static_cast<double>(n);

  auto in_edges_begin = g.in_edges.begin();

  ityr::parallel_for_each(
      {.cutoff_count = 1024, .checkout_count = 1024},
      ityr::make_global_iterator(p_curr.begin(), ityr::ori::mode::write),
      ityr::make_global_iterator(p_curr.end()  , ityr::ori::mode::write),
      [=](double& p) {
        p = one_over_n;
      });

  ityr::parallel_for_each(
      {.cutoff_count = 1024, .checkout_count = 1024},
      ityr::make_global_iterator(p_next.begin(), ityr::ori::mode::write),
      ityr::make_global_iterator(p_next.end()  , ityr::ori::mode::write),
      [=](double& p) {
        p = 0;
      });

  ityr::parallel_for_each(
      {.cutoff_count = 1024, .checkout_count = 1024},
      ityr::make_global_iterator(p_div.begin()       , ityr::ori::mode::write),
      ityr::make_global_iterator(p_div.end()         , ityr::ori::mode::write),
      ityr::make_global_iterator(g.v_out_data.begin(), ityr::ori::mode::read),
      [=](double& p, const vertex_data& vout) {
        p = one_over_n / static_cast<double>(vout.degree);
      });

  size_t iter = 0;
  while (iter++ < max_iters) {
    ityr::parallel_for_each(
        {.cutoff_count = 1, .checkout_count = 1},
        ityr::make_global_iterator(g.v_in_data.begin(), ityr::ori::mode::no_access),
        ityr::make_global_iterator(g.v_in_data.end()  , ityr::ori::mode::no_access),
        ityr::make_global_iterator(p_next.begin()     , ityr::ori::mode::no_access),
        [=](auto vin_, auto pn_) {
          vertex_data vin = vin_;

          neighbors nghs = {in_edges_begin + vin.offset, vin.degree};

          double contribution =
            ityr::parallel_reduce(
                {.cutoff_count = 1024, .checkout_count = 1024},
                nghs.begin(), nghs.end(),
                double(0), std::plus<double>{},
                [=](const uintE& idx) {
                  return p_div[idx];
                });

          pn_ = damping * contribution + addedConstant;
        });

    ityr::parallel_for_each(
        {.cutoff_count = 1024, .checkout_count = 1024},
        ityr::make_global_iterator(g.v_out_data.begin(), ityr::ori::mode::read),
        ityr::make_global_iterator(g.v_out_data.end()  , ityr::ori::mode::read),
        ityr::make_global_iterator(p_next.begin()      , ityr::ori::mode::read),
        ityr::make_global_iterator(p_div_next.begin()  , ityr::ori::mode::write),
        [=](const vertex_data& vout, const double& pn, double& pdn) {
          pdn = pn / static_cast<double>(vout.degree);
        });

    ityr::parallel_for_each(
        {.cutoff_count = 1024, .checkout_count = 1024},
        ityr::make_global_iterator(p_curr.begin()     , ityr::ori::mode::read),
        ityr::make_global_iterator(p_curr.end()       , ityr::ori::mode::read),
        ityr::make_global_iterator(p_next.begin()     , ityr::ori::mode::read),
        ityr::make_global_iterator(differences.begin(), ityr::ori::mode::write),
        [=](const double& pc, const double& pn, double& d) {
          d = fabs(pc - pn);
        });

    double L1_norm =
      ityr::parallel_reduce(
          {.cutoff_count = 1024, .checkout_count = 1024},
          ityr::make_global_iterator(differences.begin(), ityr::ori::mode::read),
          ityr::make_global_iterator(differences.end()  , ityr::ori::mode::read),
          double(0), std::plus<double>{});

    if (L1_norm < eps) break;

    /* std::cout << "L1_norm = " << L1_norm << std::endl; */

    std::swap(p_curr, p_next);
    std::swap(p_div, p_div_next);
  }

  double max_pr =
    ityr::parallel_reduce(
        {.cutoff_count = 1024, .checkout_count = 1024},
        ityr::make_global_iterator(p_next.begin(), ityr::ori::mode::read),
        ityr::make_global_iterator(p_next.end()  , ityr::ori::mode::read),
        std::numeric_limits<double>::lowest(),
        [](double a, double b) { return std::max(a, b); });

  std::cout << "max_pr = " << max_pr << " iter = " << iter << std::endl;
}

void run() {
  static std::optional<graph> g = load_dataset(dataset_filename);

  ityr::global_vector<double> p_curr     (global_vec_coll_opts, g->n);
  ityr::global_vector<double> p_next     (global_vec_coll_opts, g->n);
  ityr::global_vector<double> p_div      (global_vec_coll_opts, g->n);
  ityr::global_vector<double> p_div_next (global_vec_coll_opts, g->n);
  ityr::global_vector<double> differences(global_vec_coll_opts, g->n);

  for (int r = 0; r < n_repeats; r++) {
    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    ityr::root_exec([&]{
      pagerank(*g, {p_curr.data()     , p_curr.size()     },
                   {p_next.data()     , p_next.size()     },
                   {p_div.data()      , p_div.size()      },
                   {p_div_next.data() , p_div_next.size() },
                   {differences.data(), differences.size()});
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
