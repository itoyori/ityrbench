#pragma once

#include <optional>
#include <utility>

#include "uth.h"

#include "ityr/util.hpp"
#include "ityr/wallclock.hpp"
#include "ityr/iterator.hpp"
#include "ityr/iro.hpp"
#include "ityr/iro_ref.hpp"
#include "ityr/iro_context.hpp"
#include "ityr/ito_group.hpp"
#include "ityr/ito_pattern.hpp"
#include "ityr/logger/logger.hpp"
#include "ityr/container.hpp"

namespace ityr {

// ityr interface
// -----------------------------------------------------------------------------

template <typename P>
class ityr_if {
  struct iro_policy : public iro_policy_default {
    template <typename P_>
    using iro_impl_t = typename P::template iro_t<P_>;
    struct iro_ref_policy { using iro = iro_if<iro_policy>; };
    template <typename GPtrT>
    using global_ref = iro_ref<iro_ref_policy, GPtrT>;
    using wallclock_t = typename P::wallclock_t;
    template <typename P_>
    using logger_impl_t = typename P::template logger_impl_t<P_>;
    static constexpr bool enable_acquire_whitelist = P::enable_acquire_whitelist;
  };
  using iro_ = iro_if<iro_policy>;

  struct iro_context_policy : public iro_context_policy_default {
    template <typename P_>
    using iro_context_impl_t = typename P::template iro_context_t<P_>;
    using iro = iro_;
  };
  using iro_context_ = iro_context_if<iro_context_policy>;

  struct logger_policy : public logger_policy_default {
    template <typename P_>
    using logger_impl_t = typename P::template logger_impl_t<P_>;
    using iro = iro_;
    using wallclock_t = typename P::wallclock_t;
    using logger_kind_t = typename P::logger_kind_t;
    static const char* outfile_prefix() { return "ityr"; }
  };
  using logger_ = typename logger::template logger_if<logger_policy>;

  struct ito_group_policy : public ito_group_policy_default {
    template <typename P_, std::size_t MaxTasks, bool SpawnLastTask>
    using ito_group_impl_t = typename P::template ito_group_t<P_, MaxTasks, SpawnLastTask>;
    using iro = iro_;
    static int rank() { return P::rank(); }
    static int n_ranks() { return P::n_ranks(); }
  };
  template <std::size_t MaxTasks, bool SpawnLastTask>
  using ito_group_ = ito_group_if<ito_group_policy, MaxTasks, SpawnLastTask>;

  struct ito_pattern_policy : public ito_pattern_policy_default {
    template <typename P_>
    using ito_pattern_impl_t = typename P::template ito_pattern_t<P_>;
    using iro = iro_;
    using iro_context = iro_context_;
    static int rank() { return P::rank(); }
    static int n_ranks() { return P::n_ranks(); }
    static void barrier() { iro::release(); P::barrier(); iro::acquire(); }
    static constexpr bool auto_checkout = P::auto_checkout;
  };
  using ito_pattern_ = ito_pattern_if<ito_pattern_policy>;

  struct global_container_policy : public global_container_policy_default {
    using iro = iro_;
    using iro_context = iro_context_;
    using ito_pattern = ito_pattern_;
  };
  using global_container_ = global_container_if<global_container_policy>;

public:
  using wallclock = typename P::wallclock_t;
  using iro = iro_;
  using iro_context = iro_context_;
  template <std::size_t MaxTasks, bool SpawnLastTask = false>
  using ito_group = ito_group_<MaxTasks, SpawnLastTask>;
  using ito_pattern = ito_pattern_;
  using logger_kind = typename P::logger_kind_t::value;
  using logger = logger_;
  template <typename T>
  using global_ptr = typename iro::template global_ptr<T>;
  template <typename T>
  using global_span = typename global_container_::template global_span<T>;
  template <typename T>
  using global_vector = typename global_container_::template global_vector<T>;

  using access_mode = typename iro::access_mode;

  static_assert(!is_const_iterator_v<global_ptr<int>>);
  static_assert(is_const_iterator_v<global_ptr<const int>>);

  static int rank() { return P::rank(); }
  static int n_ranks() { return P::n_ranks(); }

  template <typename F, typename... Args>
  static void main(F f, Args... args) {
    set_signal_handlers();
    P::main(f, args...);
  }

  static void barrier() {
    iro::release();
    P::barrier();
    iro::acquire();
  }

  template <access_mode Mode, typename... Args>
  static auto with_checkout(Args&&... args) {
    return iro_context::template with_checkout<Mode>(std::forward<Args>(args)...);
  }

  template <access_mode Mode1, access_mode Mode2, typename... Args>
  static auto with_checkout(Args&&... args) {
    return iro_context::template with_checkout<Mode1, Mode2>(std::forward<Args>(args)...);
  }

  template <access_mode Mode1, access_mode Mode2, access_mode Mode3, typename... Args>
  static auto with_checkout(Args&&... args) {
    return iro_context::template with_checkout<Mode1, Mode2, Mode3>(std::forward<Args>(args)...);
  }

  template <access_mode Mode, typename... Args>
  static auto with_checkout_tied(Args&&... args) {
    return iro_context::template with_checkout_tied<Mode>(std::forward<Args>(args)...);
  }

  template <access_mode Mode1, access_mode Mode2, typename... Args>
  static auto with_checkout_tied(Args&&... args) {
    return iro_context::template with_checkout_tied<Mode1, Mode2>(std::forward<Args>(args)...);
  }

  template <access_mode Mode1, access_mode Mode2, access_mode Mode3, typename... Args>
  static auto with_checkout_tied(Args&&... args) {
    return iro_context::template with_checkout_tied<Mode1, Mode2, Mode3>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto with_checkout_cancel(Args&&... args) {
    return iro_context::with_checkout_cancel(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto root_spawn(Args&&... args) {
    return ito_pattern::root_spawn(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto master_do(Args&&... args) {
    return ito_pattern::master_do(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto parallel_invoke(Args&&... args) {
    return ito_pattern::parallel_invoke(std::forward<Args>(args)...);
  }

  template <access_mode Mode, typename... Args>
  static auto serial_for(Args&&... args) {
    return ito_pattern::template serial_for<Mode>(std::forward<Args>(args)...);
  }

  template <access_mode Mode1, access_mode Mode2, typename... Args>
  static auto serial_for(Args&&... args) {
    return ito_pattern::template serial_for<Mode1, Mode2>(std::forward<Args>(args)...);
  }

  template <access_mode Mode, typename... Args>
  static auto parallel_for(Args&&... args) {
    return ito_pattern::template parallel_for<Mode>(std::forward<Args>(args)...);
  }

  template <access_mode Mode1, access_mode Mode2, typename... Args>
  static auto parallel_for(Args&&... args) {
    return ito_pattern::template parallel_for<Mode1, Mode2>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto parallel_reduce(Args&&... args) {
    return ito_pattern::parallel_reduce(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto parallel_transform(Args&&... args) {
    return ito_pattern::parallel_transform(std::forward<Args>(args)...);
  }
};

// Serial
// -----------------------------------------------------------------------------

struct ityr_policy_serial {
  template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
  using ito_group_t = ito_group_serial<P, MaxTasks, SpawnLastTask>;

  template <typename P>
  using ito_pattern_t = ito_pattern_serial<P>;

  template <typename P>
  using iro_t = iro_dummy<P>;

  template <typename P>
  using iro_context_t = iro_context_disabled<P>;

  using wallclock_t = wallclock_native;

  using logger_kind_t = logger::kind_dummy;

  template <typename P>
  using logger_impl_t = logger::impl_dummy<P>;

  static int rank() { return 0; }

  static int n_ranks() { return 1; }

  template <typename Fn, typename... Args>
  static void main(Fn&& f, Args&&... args) { f(std::forward<Args>(args)...); }

  static void barrier() {}

  static constexpr bool auto_checkout = true;

  static constexpr bool enable_acquire_whitelist = false;
};

// Naive
// -----------------------------------------------------------------------------

struct ityr_policy_naive {
  template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
  using ito_group_t = ito_group_naive<P, MaxTasks, SpawnLastTask>;

  template <typename P>
  using ito_pattern_t = ito_pattern_naive<P>;

#ifndef ITYR_IRO_DISABLE_CACHE
#define ITYR_IRO_DISABLE_CACHE 0
#endif
#ifndef ITYR_IRO_GETPUT
#define ITYR_IRO_GETPUT 0
#endif

#if ITYR_IRO_DISABLE_CACHE
  template <typename P>
  using iro_t = iro_pcas_nocache<P>;
#elif ITYR_IRO_GETPUT
  template <typename P>
  using iro_t = iro_pcas_getput<P>;
#else
  template <typename P>
  using iro_t = iro_pcas_default<P>;
#endif

#undef ITYR_IRO_DISABLE_CACHE
#undef ITYR_IRO_GETPUT

  template <typename P>
  using iro_context_t = iro_context_enabled<P>;

  using wallclock_t = wallclock_madm;

  using logger_kind_t = logger::kind_dummy;

#ifndef ITYR_LOGGER_IMPL
#define ITYR_LOGGER_IMPL impl_dummy
#endif
  template <typename P>
  using logger_impl_t = logger::ITYR_LOGGER_IMPL<P>;
#undef ITYR_LOGGER_IMPL

  static int rank() {
    return madm::uth::get_pid();
  }

  static int n_ranks() {
    return madm::uth::get_n_procs();
  }

  template <typename Fn, typename... Args>
  static void main(Fn&& f, Args&&... args) {
    madm::uth::start(std::forward<Fn>(f), std::forward<Args>(args)...);
  }

  static void barrier() {
    madm::uth::barrier();
  }

#ifndef ITYR_AUTO_CHECKOUT
#define ITYR_AUTO_CHECKOUT true
#endif
  static constexpr bool auto_checkout = ITYR_AUTO_CHECKOUT;
#undef ITYR_AUTO_CHECKOUT

#ifndef ITYR_ENABLE_ACQUIRE_WHITELIST
#define ITYR_ENABLE_ACQUIRE_WHITELIST false
#endif
  static constexpr bool enable_acquire_whitelist = ITYR_ENABLE_ACQUIRE_WHITELIST;
#undef ITYR_AUTO_CHECKOUT
};

// Work-first fence elimination
// -----------------------------------------------------------------------------

struct ityr_policy_workfirst : public ityr_policy_naive {
  template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
  using ito_group_t = ito_group_workfirst<P, MaxTasks, SpawnLastTask>;

  template <typename P>
  using ito_pattern_t = ito_pattern_workfirst<P>;
};

// Work-first fence elimination + lazy release
// -----------------------------------------------------------------------------

struct ityr_policy_workfirst_lazy : public ityr_policy_naive {
  template <typename P, std::size_t MaxTasks, bool SpawnLastTask>
  using ito_group_t = ito_group_workfirst_lazy<P, MaxTasks, SpawnLastTask>;

  template <typename P>
  using ito_pattern_t = ito_pattern_workfirst_lazy<P>;
};

// Policy selection
// -----------------------------------------------------------------------------

#ifndef ITYR_POLICY
#define ITYR_POLICY ityr_policy_naive
#endif

using ityr_policy = ITYR_POLICY;

#undef ITYR_POLICY

}
