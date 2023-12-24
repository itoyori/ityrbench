/*
 * Heat diffusion (Jacobi-type iteration)
 *
 * Usage: see function usage();
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

#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include <errno.h>
#include <vector>
#include <algorithm>
#include <ctime>
#include <unistd.h>
#include <clocale>
#include <cstdint>
#include <mpi.h>
#include <omp.h>

/* Define ERROR_SUMMARY if you want to check your numerical results */
#define ERROR_SUMMARY

#define f(x,y)     (sin(x)*sin(y))
#define randa(x,t) (0.0)
#define randb(x,t) (exp(-2*(t))*sin(x))
#define randc(y,t) (0.0)
#define randd(y,t) (exp(-2*(t))*sin(y))
#define solu(x,y,t) (exp(-2*(t))*sin(x)*sin(y))

long nx = 4096;
long ny = 4096;
long nt = 100;
double xu = 0.0;
double xo = 157.0796326794896558;
double yu = 0.0;
double yo = 157.0796326794896558;
double tu = 0.0;
double to = 0.0000001;

double dx, dy, dt;

double dtdxsq, dtdysq;
double t;

int my_rank, n_ranks;

inline uint64_t clock_gettime_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000 + (uint64_t)ts.tv_nsec;
}

/*****************   Initialization of grid partition  ********************/

void initgrid(double *old_a, long lb, long ub) {
#pragma omp parallel for
  for (long a=lb+1; a < ub-1; a++) {	/* inner nodes */
    for (long b=1; b < ny-1; b++) {
      old_a[(a - lb) * ny + b] = f(xu + a * dx, yu + b * dy);
    }
  }

#pragma omp parallel for
  for (long a=lb+1; a < ub-1; a++)		/* boundary nodes */
    old_a[(a - lb) * ny + 0] = randa(xu + a * dx, 0);

#pragma omp parallel for
  for (long a=lb+1; a < ub-1; a++)
    old_a[(a - lb) * ny + ny-1] = randb(xu + a * dx, 0);

  if (lb == 0) {
#pragma omp parallel for
    for (long b=0; b < ny; b++)
      old_a[b] = randc(yu + b * dy, 0);
  }
  if (ub == nx) {
#pragma omp parallel for
    for (long b=0; b < ny; b++)
      old_a[(nx - 1 - lb) * ny + b] = randd(yu + b * dy, 0);
  }
}


/***************** Five-Point-Stencil Computation ********************/

void compstripe(double *new_a, double *old_a, long lb, long ub) {
#pragma omp parallel for
  for (long a=lb+1; a < ub-1; a++) {
    for (int b=1; b < ny-1; b++) {
      int64_t s = (a - lb) * ny + b;
      new_a[s] = dtdxsq * (old_a[s + ny] - 2 * old_a[s] + old_a[s - ny])
	       + dtdysq * (old_a[s + 1] - 2 * old_a[s] + old_a[s - 1])
  	       + old_a[s];
    }
  }

#pragma omp parallel for
  for (long a=lb+1; a < ub-1; a++)
    new_a[(a - lb) * ny + ny-1] = randb(xu + a * dx, t);

#pragma omp parallel for
  for (long a=lb+1; a < ub-1; a++)
    new_a[(a - lb) * ny + 0] = randa(xu + a * dx, t);

  if (lb == 0) {
#pragma omp parallel for
    for (long b=0; b < ny; b++)
      new_a[b] = randc(yu + b * dy, t);
  }
  if (ub == nx) {
#pragma omp parallel for
    for (long b=0; b < ny; b++)
      new_a[(nx - 1 - lb) * ny + b] = randd(yu + b * dy, t);
  }
}


/***************** Decomposition of 2D grids in stripes ********************/

#define INIT       1
#define COMP       2

int divide(long lb, long ub, double *new_a, double *old_a, int mode, long timestep) {
  if (mode == INIT) {
    initgrid(old_a, lb, ub);
    return 1;
  }

  if (timestep % 2 == 0)
    std::swap(new_a, old_a);

  std::vector<MPI_Request> reqs;

  if (my_rank != 0) {
    MPI_Request req1, req2;
    MPI_Isend(&old_a[ny], ny, MPI_DOUBLE, my_rank - 1, timestep, MPI_COMM_WORLD, &req1);
    MPI_Irecv(&old_a[ 0], ny, MPI_DOUBLE, my_rank - 1, timestep, MPI_COMM_WORLD, &req2);
    reqs.push_back(req1);
    reqs.push_back(req2);
  }

  if (my_rank != n_ranks - 1) {
    MPI_Request req1, req2;
    MPI_Isend(&old_a[(ub - lb - 2) * ny], ny, MPI_DOUBLE, my_rank + 1, timestep, MPI_COMM_WORLD, &req1);
    MPI_Irecv(&old_a[(ub - lb - 1) * ny], ny, MPI_DOUBLE, my_rank + 1, timestep, MPI_COMM_WORLD, &req2);
    reqs.push_back(req1);
    reqs.push_back(req2);
  }

  MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

  compstripe(new_a, old_a, lb, ub);

  MPI_Barrier(MPI_COMM_WORLD);

  return 1;
}

int heat() {
  long lb = std::max((my_rank * nx) / n_ranks - 1, 0L);
  long ub = std::min(((my_rank + 1) * nx) / n_ranks + 1, nx);

  /* Memory Allocation */
  double* old_a = (double *) malloc((ub - lb) * ny * sizeof(double));
  double* new_a = (double *) malloc((ub - lb) * ny * sizeof(double));

  /* Initialization */
  divide(lb, ub, new_a, old_a, INIT, 0);

  /* Jacobi Iteration (divide x-dimension of 2D grid into stripes) */
  /* Timing. "Start" timers */

  for (long c = 1; c <= nt; c++) {
    t = tu + c * dt;

    auto t0 = clock_gettime_ns();

    divide(lb, ub, new_a, old_a, COMP, c);

    auto t1 = clock_gettime_ns();

    if (my_rank == 0) {
      printf("[%ld] %ld ns\n", c, t1 - t0);
      fflush(stdout);
    }
  }

#ifdef ERROR_SUMMARY
  /* Error summary computation: Not parallelized! */
  auto llb = (lb == 0) ? 0 : lb + 1;
  auto lub = (ub == nx) ? nx : ub - 1;
  auto mat = (nt % 2) ? new_a : old_a;
  if (my_rank == 0) {
    printf("\n Error summary of last time frame comparing with exact solution:");
  }
  double mae = 0.0;
#pragma omp parallel for reduction(max:mae)
  for (long a=llb; a<lub; a++)
    for (long b=0; b<ny; b++) {
      double tmp = fabs(mat[(a-lb)*ny+b] - solu(xu + a * dx, yu + b * dy, to));
      if (tmp > mae)
	mae = tmp;
    }
  double mae_all;
  MPI_Reduce(&mae, &mae_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (my_rank == 0) {
    printf("\n   Local maximal absolute error  %10e ", mae_all);
  }

  double mre = 0.0;
#pragma omp parallel for reduction(max:mre)
  for (long a=llb; a<lub; a++)
    for (long b=0; b<ny; b++) {
      double tmp = fabs(mat[(a-lb)*ny+b] - solu(xu + a * dx, yu + b * dy, to));
      if (mat[(a-lb)*ny+b] != 0.0)
	tmp = tmp / mat[(a-lb)*ny+b];
      if (tmp > mre)
	mre = tmp;
    }
  double mre_all;
  MPI_Reduce(&mre, &mre_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (my_rank == 0) {
    printf("\n   Local maximal relative error  %10e %s ", mre_all * 100, "%");
  }

  double me = 0.0;
#pragma omp parallel for reduction(+:me)
  for (long a=llb; a<lub; a++)
    for (long b=0; b<ny; b++) {
      me += fabs(mat[(a-lb)*ny+b] - solu(xu + a * dx, yu + b * dy, to));
    }
  double me_all;
  MPI_Reduce(&me, &me_all, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (my_rank == 0) {
    me_all = me_all / (nx * ny);
    printf("\n   Global Mean absolute error    %10e\n\n", me_all);
  }
#endif

  free(new_a);
  free(old_a);
  return 0;
}

void show_help_and_exit(int argc [[maybe_unused]], char** argv) {
  if (my_rank == 0) {
    printf("Usage: %s [options]\n"
           "  options:\n"
           "    -x : total number of columns (long)\n"
           "    -y : total number of rows (long)\n"
           "    -t : total time steps (long)\n", argv[0]);
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

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  int opt;
  while ((opt = getopt(argc, argv, "x:y:t:h")) != EOF) {
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

  int n_omp_threads = 1;
#pragma omp parallel
  {
#pragma omp single
    {
      n_omp_threads = omp_get_num_threads();
    }
  }

  if (my_rank == 0) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[Heat]\n"
           "# of processes:               %d\n"
           "# of threads:                 %d\n"
           "nx:                           %ld\n"
           "ny:                           %ld\n"
           "nt:                           %ld\n"
           "dx:                           %.10f\n"
           "dy:                           %.10f\n"
           "dt:                           %.10f\n"
           "-------------------------------------------------------------\n",
           n_ranks, n_omp_threads, nx, ny, nt, dx, dy, dt);

    printf("=============================================================\n");
    printf("PID of the main worker: %d\n", getpid());
    printf("\n\n Stability Value for explicit method must be > 0:  %f\n\n",
           0.5 - (dt / (dx * dx) + dt / (dy * dy)));
    fflush(stdout);
  }

  heat();

  MPI_Finalize();
  return 0;
}
