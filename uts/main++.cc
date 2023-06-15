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
#include <new>

#include "../common.hpp"

template <typename T>
using global_ptr = ityr::ori::global_ptr<T>;

#include "uts.h"

#ifndef UTS_USE_VECTOR
#define UTS_USE_VECTOR 0
#endif

#ifndef UTS_REBUILD_TREE
#define UTS_REBUILD_TREE 0
#endif

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

#if UTS_USE_VECTOR

struct dynamic_node {
  int n_children;
  ityr::global_vector<global_ptr<dynamic_node>> children;
};

global_ptr<dynamic_node> new_dynamic_node(int n_children) {
  auto gptr = ityr::ori::malloc<dynamic_node>(1);
  auto cs = ityr::make_checkout(gptr, 1, ityr::checkout_mode::write);
  dynamic_node& new_node = *(new (&cs[0]) dynamic_node);
  new_node.n_children = n_children;
  new_node.children.resize(n_children);
  return gptr;
}

global_ptr<global_ptr<dynamic_node>> get_children(global_ptr<dynamic_node> node) {
  auto cs = ityr::make_checkout(node, 1, ityr::checkout_mode::read);
  return cs[0].children.data();
}

void delete_dynamic_node(global_ptr<dynamic_node> node, int) {
  {
    auto cs = ityr::make_checkout(node, 1, ityr::checkout_mode::read_write);
    std::destroy_at(&cs[0]);
  }
  ityr::ori::free(node, 1);
}

#else

struct dynamic_node {
  int n_children;
  global_ptr<dynamic_node> children[1];
};

std::size_t node_size(int n_children) {
  return sizeof(dynamic_node) + (n_children - 1) * sizeof(global_ptr<dynamic_node>);
}

global_ptr<dynamic_node> new_dynamic_node(int n_children) {
  auto gptr = ityr::ori::reinterpret_pointer_cast<dynamic_node>(
      ityr::ori::malloc<std::byte>(node_size(n_children)));
  gptr->*(&dynamic_node::n_children) = n_children;
  return gptr;
}

void delete_dynamic_node(global_ptr<dynamic_node> node, int n_children) {
  ityr::ori::free(ityr::ori::reinterpret_pointer_cast<std::byte>(node), node_size(n_children));
}

global_ptr<global_ptr<dynamic_node>> get_children(global_ptr<dynamic_node> node) {
  return global_ptr<global_ptr<dynamic_node>>(&(node->*(&dynamic_node::children)));
}

#endif

global_ptr<dynamic_node> build_tree(Node parent) {
  counter_t numChildren = uts_numChildren(&parent);
  int childType = uts_childType(&parent);

  global_ptr<dynamic_node> this_node = new_dynamic_node(numChildren);
  global_ptr<global_ptr<dynamic_node>> children = get_children(this_node);

  if (numChildren > 0) {
    ityr::parallel_for_each(
        ityr::count_iterator<counter_t>(0),
        ityr::count_iterator<counter_t>(numChildren),
        ityr::make_global_iterator(children, ityr::checkout_mode::no_access),
        [=](counter_t i, auto&& x) {
          Node child = makeChild(&parent, childType,
                                 computeGranularity, i);
          x = build_tree(child);
        });
  }

  return this_node;
}

Result traverse_tree(counter_t depth, global_ptr<dynamic_node> this_node) {
  counter_t numChildren = this_node->*(&dynamic_node::n_children);
  global_ptr<global_ptr<dynamic_node>> children = get_children(this_node);

  if (numChildren == 0) {
    return { depth, 1, 1 };
  } else {
    Result result = ityr::parallel_reduce(
        ityr::make_global_iterator(children              , ityr::checkout_mode::no_access),
        ityr::make_global_iterator(children + numChildren, ityr::checkout_mode::no_access),
        Result{0, 0, 0},
        mergeResult,
        [=](auto&& child_node) {
          return traverse_tree(depth + 1, child_node);
        });
    result.size += 1;
    return result;
  }
}

void destroy_tree(global_ptr<dynamic_node> this_node) {
  counter_t numChildren = this_node->*(&dynamic_node::n_children);
  global_ptr<global_ptr<dynamic_node>> children = get_children(this_node);

  if (numChildren > 0) {
    ityr::parallel_for_each(
        ityr::make_global_iterator(children              , ityr::checkout_mode::no_access),
        ityr::make_global_iterator(children + numChildren, ityr::checkout_mode::no_access),
        [=](auto&& child_node) {
          destroy_tree(child_node);
        });
  }

  delete_dynamic_node(this_node, numChildren);
}

//-- main ---------------------------------------------------------------------

void uts_run() {
  global_ptr<dynamic_node> root_node;

  for (int i = 0; i < numRepeats; i++) {
    if (UTS_REBUILD_TREE || i == 0) {
      auto t0 = ityr::gettime_ns();
      Node root;
      uts_initRoot(&root, type);
      root_node = ityr::root_exec([=]() {
        return build_tree(root);
      });
      auto t1 = ityr::gettime_ns();

      if (ityr::is_master()) {
        printf("## Tree built. (%ld ns)\n", t1 - t0);
        fflush(stdout);
      }
    }

    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    Result r = ityr::root_exec([=]() {
      return traverse_tree(0, root_node);
    });

    auto t1 = ityr::gettime_ns();

    ityr::profiler_end();

    uint64_t walltime = t1 - t0;

    counter_t maxTreeDepth = r.maxdepth;
    counter_t nNodes = r.size;
    counter_t nLeaves = r.leaves;

    double perf = (double)nNodes / walltime;

    if (ityr::is_master()) {
      printf("[%d] %ld ns %.6g Gnodes/s ( nodes: %llu depth: %llu leaves: %llu )\n",
             i, walltime, perf, nNodes, maxTreeDepth, nLeaves);
      fflush(stdout);
    }

    ityr::profiler_flush();

    if (UTS_REBUILD_TREE || i == numRepeats - 1) {
      auto t0 = ityr::gettime_ns();
      ityr::root_exec([=]() {
        destroy_tree(root_node);
      });
      auto t1 = ityr::gettime_ns();

      if (ityr::is_master()) {
        printf("## Tree destroyed. (%ld ns)\n", t1 - t0);
        fflush(stdout);
      }
    }

    ityr::barrier();
    /* ityr::ori::collect_deallocated(); */
    /* ityr::barrier(); */
  }
}

int main(int argc, char** argv) {
  ityr::init();
  set_signal_handlers();

  uts_parseParams(argc, argv);

  if (ityr::is_master()) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[UTS++]\n"
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

  uts_run();

  ityr::fini();
  return 0;
}
