/*
 * Heat diffusion (Jacobi-type iteration)
 *
 * Volker Strumpen, Boston                                 August 1996
 *
 * Copyright (c) 1996 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "common.hpp"

/* Define ERROR_SUMMARY if you want to check your numerical results */
#define ERROR_SUMMARY

#define f(x,y)     (sin(x)*sin(y))
#define randa(x,t) (0.0)
#define randb(x,t) (exp(-2*(t))*sin(x))
#define randc(y,t) (0.0)
#define randd(y,t) (exp(-2*(t))*sin(y))
#define solu(x,y,t) (exp(-2*(t))*sin(x)*sin(y))

long nx = 512;
long ny = 512;
long nt = 100;
double xu = 0.0;
double xo = 1.570796326794896558;
double yu = 0.0;
double yo = 1.570796326794896558;
double tu = 0.0;
double to = 0.0000001;

double dx, dy, dt;

double dtdxsq, dtdysq;
double t;

long leafmaxcol = 10;

/*****************   Initialization of grid partition  ********************/

void initgrid(ityr::global_span<double> old_a, long lb, long ub) {
  long llb = (lb == 0) ? 1 : lb;
  long lub = (ub == nx) ? nx - 1 : ub;

  if (lub > llb) {
    auto old_cs = ityr::make_checkout(old_a.subspan(llb * ny, (lub - llb) * ny), ityr::checkout_mode::write);

    for (long a = 0; a < lub - llb; a++) {	/* inner nodes */
      for (long b = 1; b < ny - 1; b++) {
        old_cs[a * ny + b] = f(xu + (a + llb) * dx, yu + b * dy);
      }
    }

    for (long a = 0; a < lub - llb; a++)		/* boundary nodes */
      old_cs[a * ny] = randa(xu + (a + llb) * dx, 0);

    for (long a = 0; a < lub - llb; a++)
      old_cs[a * ny + ny - 1] = randb(xu + (a + llb) * dx, 0);
  }

  if (lb == 0) {
    auto old_cs = ityr::make_checkout(old_a.subspan(0, ny), ityr::checkout_mode::write);
    for (long b = 0; b < ny; b++)
      old_cs[b] = randc(yu + b * dy, 0);
  }

  if (ub == nx) {
    auto old_cs = ityr::make_checkout(old_a.subspan((nx - 1) * ny, ny), ityr::checkout_mode::write);
    for (long b = 0; b < ny; b++)
      old_cs[b] = randd(yu + b * dy, 0);
  }
}


/***************** Five-Point-Stencil Computation ********************/

void compstripe(ityr::global_span<double> new_a, ityr::global_span<double> old_a, long lb, long ub) {
  long llb = (lb == 0) ? 1 : lb;
  long lub = (ub == nx) ? nx - 1 : ub;

  if (lub > llb) {
    auto [new_cs, old_cs] = ityr::make_checkouts(
        new_a.subspan(llb       * ny, (lub - llb)     * ny), ityr::checkout_mode::read_write,
        old_a.subspan((llb - 1) * ny, (lub - llb + 2) * ny), ityr::checkout_mode::read);

    for (long a = 0; a < lub - llb; a++) {
      for (long b = 1; b < ny - 1; b++) {
        new_cs[a * ny + b] = dtdxsq * (old_cs[(a + 2) * ny + b    ] - 2 * old_cs[(a + 1) * ny + b] + old_cs[      a * ny + b    ])
                           + dtdysq * (old_cs[(a + 1) * ny + b + 1] - 2 * old_cs[(a + 1) * ny + b] + old_cs[(a + 1) * ny + b - 1])
                           + old_cs[(a + 1) * ny + b];
      }
    }

    for (long a = 0; a < lub - llb; a++)
      new_cs[a * ny + ny - 1] = randb(xu + (a + llb) * dx, t);

    for (long a = 0; a < lub - llb; a++)
      new_cs[a * ny] = randa(xu + (a + llb) * dx, t);
  }

  if (lb == 0) {
    auto new_cs = ityr::make_checkout(new_a.subspan(0, ny), ityr::checkout_mode::read_write);
    for (long b = 0; b < ny; b++)
      new_cs[b] = randc(yu + b * dy, t);
  }

  if (ub == nx) {
    auto new_cs = ityr::make_checkout(new_a.subspan((nx - 1) * ny, ny), ityr::checkout_mode::read_write);
    for (long b = 0; b < ny; b++)
      new_cs[b] = randd(yu + b * dy, t);
  }
}


/***************** Decomposition of 2D grids in stripes ********************/

#define INIT       1
#define COMP       2

long divide(long lb, long ub, ityr::global_span<double> new_a, ityr::global_span<double> old_a, long mode, long timestep) {
  if (ub - lb > leafmaxcol) {
    auto [l, r] =
      ityr::parallel_invoke(
          [=] { return divide(lb, (ub + lb) / 2, new_a, old_a, mode, timestep); },
          [=] { return divide((ub + lb) / 2, ub, new_a, old_a, mode, timestep); });
    return l + r;

  } else if (ub > lb) {
    switch (mode) {
    case COMP:
      if (timestep % 2)
	compstripe(new_a, old_a, lb, ub);
      else
	compstripe(old_a, new_a, lb, ub);
      return 1;

    case INIT:
      initgrid(old_a, lb, ub);
      return 1;
    }
  }

  return 0;
}


long heat() {
  /* Memory Allocation */
  ityr::global_vector<double> new_v(ityr::global_vector_options(true, 1024), nx * ny);
  ityr::global_vector<double> old_v(ityr::global_vector_options(true, 1024), nx * ny);

  ityr::global_span<double> new_a(new_v);
  ityr::global_span<double> old_a(old_v);

  /* Initialization */
  ityr::root_exec([=] {
    divide(0, nx, new_a, old_a, INIT, 0);
  });

  /* Jacobi Iteration (divide x-dimension of 2D grid into stripes) */
  for (long c = 1; c <= nt; c++) {
    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    t = tu + c * dt;
    ityr::root_exec([=] {
      divide(0, nx, new_a, old_a, COMP, c);
    });

    auto t1 = ityr::gettime_ns();

    ityr::profiler_end();

    if (ityr::is_master()) {
      printf("[%ld] %ld ns\n", c, t1 - t0);
      fflush(stdout);
    }

    ityr::profiler_flush();
  }

#ifdef ERROR_SUMMARY
  auto mat = (nt % 2) ? new_a : old_a;

  ityr::root_exec([=] {
    ityr::migrate_to_master();
    printf("\n Error summary of last time frame comparing with exact solution:");

    double mae = ityr::transform_reduce(
        ityr::execution::parallel_policy(1024),
        mat.begin(), mat.end(), ityr::count_iterator<long>(0),
        ityr::reducer::max<double>{},
        [=](const double& r, long idx) {
          long a = idx / ny;
          long b = idx % ny;
          return fabs(r - solu(xu + a * dx, yu + b * dy, to));
        });
    ityr::migrate_to_master();
    printf("\n   Local maximal absolute error  %10e ", mae);

    double mre = ityr::transform_reduce(
        ityr::execution::parallel_policy(1024),
        mat.begin(), mat.end(), ityr::count_iterator<long>(0),
        ityr::reducer::max<double>{},
        [=](const double& r, long idx) {
          long a = idx / ny;
          long b = idx % ny;
          double tmp = fabs(r - solu(xu + a * dx, yu + b * dy, to));
          if (r != 0.0)
            tmp = tmp / r;
          return tmp;
        });
    ityr::migrate_to_master();
    printf("\n   Local maximal relative error  %10e %s ", mre * 100, "%");

    double me = ityr::transform_reduce(
        ityr::execution::parallel_policy(1024),
        mat.begin(), mat.end(), ityr::count_iterator<long>(0),
        ityr::reducer::plus<double>{},
        [=](const double& r, long idx) {
          long a = idx / ny;
          long b = idx % ny;
          return fabs(r - solu(xu + a * dx, yu + b * dy, to));
        });
    me = me / (nx * ny);
    ityr::migrate_to_master();
    printf("\n   Global Mean absolute error    %10e\n\n", me);
  });
#endif
  return 0;
}

void show_help_and_exit(int argc [[maybe_unused]], char** argv) {
  if (ityr::is_master()) {
    printf("Usage: %s [options]\n"
           "  options:\n"
           "    -x : total number of columns (long)\n"
           "    -y : total number of rows (long)\n"
           "    -t : total time steps (long)\n"
           "    -c : task granularity, columns per partition (long)\n", argv[0]);
  }
  /*
    fprintf(stderr, "   -xu #    lower x coordinate default: 0.0\n");
    fprintf(stderr, "   -xo #    upper x coordinate default: 1.570796326794896558\n");
    fprintf(stderr, "   -yu #    lower y coordinate default: 0.0\n");
    fprintf(stderr, "   -yo #    upper y coordinate default: 1.570796326794896558\n");
    fprintf(stderr, "   -tu #    start time         default: 0.0\n");
    fprintf(stderr, "   -to #    end time           default: 0.0000001\n");
   */
  exit(1);
}

int main(int argc, char** argv) {
  ityr::init();

  set_signal_handlers();

  int opt;
  while ((opt = getopt(argc, argv, "x:y:t:c:h")) != EOF) {
    switch (opt) {
      case 'x':
        nx = atoi(optarg);
        break;
      case 'y':
        ny = atoi(optarg);
        break;
      case 't':
        nt = atoi(optarg);
        break;
      case 'c':
        leafmaxcol = atoi(optarg);
        break;
      case 'h':
      default:
        show_help_and_exit(argc, argv);
    }
  }

  dx = (xo - xu) / (nx - 1);
  dy = (yo - yu) / (ny - 1);
  dt = (to - tu) / nt;	/* nt effective time steps! */

  dtdxsq = dt / (dx * dx);
  dtdysq = dt / (dy * dy);

  if (ityr::is_master()) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[Heat]\n"
           "# of processes:               %d\n"
           "nx:                           %ld\n"
           "ny:                           %ld\n"
           "nt:                           %ld\n"
           "dx:                           %.10f\n"
           "dy:                           %.10f\n"
           "dt:                           %.10f\n"
           "leafmaxcol:                   %ld\n"
           "-------------------------------------------------------------\n",
           ityr::n_ranks(), nx, ny, nt, dx, dy, dt, leafmaxcol);

    printf("[Compile Options]\n");
    ityr::print_compile_options();
    printf("-------------------------------------------------------------\n");
    printf("[Runtime Options]\n");
    ityr::print_runtime_options();
    printf("=============================================================\n");
    printf("PID of the main worker: %d\n", getpid());
    printf("\n\n Stability Value for explicit method must be > 0:  %f\n\n",
           0.5 - (dt / (dx * dx) + dt / (dy * dy)));
    fflush(stdout);
  }

  heat();

  ityr::fini();
  return 0;
}
