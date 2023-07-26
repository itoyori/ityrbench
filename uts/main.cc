/*
 *         ---- The Unbalanced Tree Search (UTS) Benchmark ----
 *  
 *  Copyright (c) 2010 See AUTHORS file for copyright holders
 *
 *  This file is part of the unbalanced tree search benchmark.  This
 *  project is licensed under the MIT Open Source license.  See the LICENSE
 *  file for copyright and licensing information.
 *
 *  UTS is a collaborative project between researchers at the University of
 *  Maryland, the University of North Carolina at Chapel Hill, and the Ohio
 *  State University.  See AUTHORS file for more information.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../common.hpp"

#include "uts.h"

#ifndef UTS_RUN_SEQ
#define UTS_RUN_SEQ 0
#endif

#ifndef UTS_RECURSIVE_FOR
#define UTS_RECURSIVE_FOR 0
#endif

/***********************************************************
 *  UTS Implementation Hooks                               *
 ***********************************************************/

// The name of this implementation
const char * impl_getName(void) {
  return "Itoyori Parallel Search";
}

int impl_paramsToStr(char *strBuf, int ind) {
  ind += sprintf(strBuf + ind, "Execution strategy:  %s\n", impl_getName());
  return ind;
}

// Not using UTS command line params, return non-success
int impl_parseParam(char *, char *) {
  return 1;
}

void impl_helpMessage(void) {
  printf("   none.\n");
}

void impl_abort(int err) {
  exit(err);
}

/***********************************************************
 * Recursive depth-first implementation                    *
 ***********************************************************/

typedef struct {
  counter_t maxdepth, size, leaves;
} Result;

Result mergeResult(Result r0, Result r1) {
  Result r = {
    (r0.maxdepth > r1.maxdepth) ? r0.maxdepth : r1.maxdepth,
    r0.size + r1.size,
    r0.leaves + r1.leaves
  };
  return r;
}

Node makeChild(const Node *parent, int childType, int computeGranularity, counter_t idx) {
  int j;

  Node c = { childType, (int)parent->height + 1, -1, {{0}} };

  for (j = 0; j < computeGranularity; j++) {
    rng_spawn(parent->state.state, c.state.state, (int)idx);
  }

  return c;
}

//-- sequential -------------------------------------------------------------

Result parTreeSearch_fj_seq(counter_t depth, Node parent);

#if UTS_RECURSIVE_FOR
Result doParTreeSearch_fj_seq(counter_t depth, Node parent, int childType,
                              counter_t numChildren, counter_t begin, counter_t end) {
  if (end - begin == 1) {
    Node child = makeChild(&parent, childType, computeGranularity, begin);
    return parTreeSearch_fj_seq(depth, child);
  } else {
    counter_t center = (begin + end) / 2;

    Result r0 = doParTreeSearch_fj_seq(depth, parent, childType,
                                       numChildren, begin, center);

    Result r1 = doParTreeSearch_fj_seq(depth, parent, childType,
                                       numChildren, center, end);
    return mergeResult(r0, r1);
  }
}
#endif

Result parTreeSearch_fj_seq(counter_t depth, Node parent) {
  Result result;

  assert(depth == 0 || parent.height > 0);

  counter_t numChildren = uts_numChildren(&parent);
  int childType = uts_childType(&parent);

  // Recurse on the children
  if (numChildren == 0) {
    Result r = { depth, 1, 1 };
    result = r;
  } else {

#if UTS_RECURSIVE_FOR
    result = doParTreeSearch_fj_seq(depth + 1, parent, childType,
                                    numChildren, 0, numChildren);
#else
    Result init = { 0, 0, 0 };
    result = init;
    for (counter_t i = 0; i < numChildren; i++) {
      Node child = makeChild(&parent, childType, computeGranularity, i);
      Result r = parTreeSearch_fj_seq(depth + 1, child);
      result = mergeResult(result, r);
    }
#endif

    result.size += 1;
  }

  return result;
}

//-- parallel ---------------------------------------------------------------

Result parTreeSearch_fj(counter_t depth, Node parent) {
  Result result;

  assert(depth == 0 || parent.height > 0);

  counter_t numChildren = uts_numChildren(&parent);
  int childType = uts_childType(&parent);

  // Recurse on the children
  if (numChildren == 0) {
    Result r = { depth, 1, 1 };
    result = r;
  } else {
    result = ityr::transform_reduce(
        ityr::execution::par,
        ityr::count_iterator<counter_t>(0),
        ityr::count_iterator<counter_t>(numChildren),
        Result{0, 0, 0},
        mergeResult,
        [=](counter_t i) {
          Node child = makeChild(&parent, childType,
                                 computeGranularity, i);
          return parTreeSearch_fj(depth + 1, child);
        });

    result.size += 1;
  }

  return result;
}

//-- main ---------------------------------------------------------------------

Result uts_fj_run(counter_t depth, Node parent) {
#if UTS_RUN_SEQ
  return parTreeSearch_fj_seq(depth, parent);
#else
  return ityr::root_exec([=]() {
    return parTreeSearch_fj(depth, parent);
  });
#endif
}

Result uts_fj_main() {
  Node root;
  uts_initRoot(&root, type);

  return uts_fj_run((counter_t)0, root);
}

int main(int argc, char** argv) {
  counter_t nNodes = 0;
  counter_t nLeaves = 0;
  counter_t maxTreeDepth = 0;

  ityr::init();
  set_signal_handlers();

  uts_parseParams(argc, argv);

  if (ityr::is_master()) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[UTS]\n"
           "# of processes:                %d\n"
           "# of repeats:                  %d\n"
           "-------------------------------------------------------------\n",
           ityr::n_ranks(), numRepeats);

    if (type == GEO) {
      printf("t (Tree type):                 Geometric (%d)\n"
             "r (Seed):                      %d\n"
             "b (Branching factor):          %f\n"
             "a (Shape function):            %d\n"
             "d (Depth):                     %d\n"
             "-------------------------------------------------------------\n",
             type, rootId, b_0, shape_fn, gen_mx);
    } else if (type == BIN) {
      printf("t (Tree type):                 Binomial (%d)\n"
             "r (Seed):                      %d\n"
             "b (# of children at root):     %f\n"
             "m (# of children at non-root): %d\n"
             "q (Prob for having children):  %f\n"
             "-------------------------------------------------------------\n",
             type, rootId, b_0, nonLeafBF, nonLeafProb);
    } else {
      assert(0); // TODO:
    }

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

  for (int i = 0; i < numRepeats; i++) {
    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    Result r = uts_fj_main();

    auto t1 = ityr::gettime_ns();

    ityr::profiler_end();

    if (ityr::is_master()) {
      maxTreeDepth = r.maxdepth;
      nNodes = r.size;
      nLeaves = r.leaves;

      double perf = (double)nNodes / (t1 - t0);

      printf("[%d] %ld ns %.6g Gnodes/s ( nodes: %llu depth: %llu leaves: %llu )\n",
             i, t1 - t0, perf, nNodes, maxTreeDepth, nLeaves);
      fflush(stdout);
    }

    ityr::profiler_flush();
  }

  ityr::fini();
  return 0;
}
