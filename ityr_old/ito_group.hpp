#pragma once

#include "uth.h"

#include "ityr/iro.hpp"

namespace ityr {

template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
class ito_group_if {
  typename P::template ito_group_impl_t<P, MaxTasks, SpawnLastTask> impl_;

public:
  ito_group_if() : impl_() {}

  template <typename Fn, typename... Args>
  void run(Fn&& f, Args&&... args) { impl_.run(std::forward<Fn>(f), std::forward<Args>(args)...); }

  void wait() { impl_.wait(); }
};

template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
class ito_group_serial {
public:
  ito_group_serial() {}

  template <typename Fn, typename... Args>
  void run(Fn&& f, Args&&... args) {
    std::forward<Fn>(f)(std::forward<Args>(args)...);
  }

  void wait() {}
};

template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
class ito_group_naive {
  using iro = typename P::iro;

  madm::uth::thread<void> tasks_[MaxTasks];
  std::size_t n_ = 0;

public:
  ito_group_naive() {}

  template <typename Fn, typename... Args>
  void run(Fn&& f, Args&&... args) {
    assert(n_ < MaxTasks);
    if (SpawnLastTask || n_ < MaxTasks - 1) {
      iro::release();
      new (&tasks_[n_++]) madm::uth::thread<void>{[=] {
        iro::acquire();
        f(args...);
        iro::release();
      }};
      iro::acquire();
    } else {
      std::forward<Fn>(f)(std::forward<Args>(args)...);
    }
  }

  void wait() {
    iro::release();
    for (std::size_t i = 0; i < n_; i++) {
      tasks_[i].join();
    }
    iro::acquire();
    n_ = 0;
  }
};

template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
class ito_group_workfirst {
  using iro = typename P::iro;

  madm::uth::thread<void> tasks_[MaxTasks];
  bool all_synched_ = true;
  int initial_rank;
  std::size_t n_ = 0;

public:
  ito_group_workfirst() { initial_rank = P::rank(); }

  template <typename Fn, typename... Args>
  void run(Fn&& f, Args&&... args) {
    iro::poll();

    assert(n_ < MaxTasks);
    if (SpawnLastTask || n_ < MaxTasks - 1) {
      auto p_th = &tasks_[n_];
      iro::release();
      new (p_th) madm::uth::thread<void>{};
      bool synched = p_th->spawn_aux(std::forward<Fn>(f),
        std::make_tuple(std::forward<Args>(args)...),
        [=] (bool parent_popped) {
          // on-die callback
          if (!parent_popped) {
            iro::release();
          }
        });
      if (!synched) {
        iro::acquire();
      }
      all_synched_ &= synched;
      n_++;
    } else {
      std::forward<Fn>(f)(std::forward<Args>(args)...);
    }

    iro::poll();
  }

  void wait() {
    bool blocked = false;
    for (std::size_t i = 0; i < n_; i++) {
      iro::poll();
      tasks_[i].join_aux(0, [&blocked] {
        // on-block callback
        if (!blocked) {
          iro::release();
          blocked = true;
        }
      });
    }
    if (initial_rank != P::rank() || !all_synched_ || blocked) {
      // FIXME: (all_synched && blocked) is true only for root tasks
      iro::acquire();
    }
    n_ = 0;

    iro::poll();
  }
};

template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
class ito_group_workfirst_lazy {
  using iro = typename P::iro;

  madm::uth::thread<void> tasks_[MaxTasks];
  bool all_synched_ = true;
  int initial_rank;
  std::size_t n_ = 0;

public:
  ito_group_workfirst_lazy() { initial_rank = P::rank(); }

  template <typename Fn, typename... Args>
  void run(Fn&& f, Args&&... args) {
    iro::poll();

    assert(n_ < MaxTasks);
    if (SpawnLastTask || n_ < MaxTasks - 1) {
      typename iro::release_handler rh;
      iro::release_lazy(&rh);

      auto p_th = &tasks_[n_];
      new (p_th) madm::uth::thread<void>{};
      bool synched = p_th->spawn_aux(std::forward<Fn>(f),
        std::make_tuple(std::forward<Args>(args)...),
        [=] (bool parent_popped) {
          // on-die callback
          if (!parent_popped) {
            iro::release();
          }
        });
      if (!synched) {
        iro::acquire(rh);
      }
      all_synched_ &= synched;
      n_++;
    } else {
      std::forward<Fn>(f)(std::forward<Args>(args)...);
    }

    iro::poll();
  }

  void wait() {
    bool blocked = false;
    for (std::size_t i = 0; i < n_; i++) {
      iro::poll();
      tasks_[i].join_aux(0, [&blocked] {
        // on-block callback
        if (!blocked) {
          iro::release();
          blocked = true;
        }
      });
    }
    if (initial_rank != P::rank() || !all_synched_ || blocked) {
      // FIXME: (all_synched && blocked) is true only for root tasks
      iro::acquire();
    }
    n_ = 0;

    iro::poll();
  }
};

struct ito_group_policy_default {
  template <typename P_, std::size_t MaxTasks, bool SpawnLastTask>
  using ito_group_impl_t = ito_group_serial<P_, MaxTasks, SpawnLastTask>;
  using iro = iro_if<iro_policy_default>;
  static int rank() { return 0; }
  static int n_ranks() { return 1; }
};

}
