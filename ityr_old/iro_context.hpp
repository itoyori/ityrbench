#pragma once

#include "pcas/pcas.hpp"

#include "ityr/util.hpp"
#include "ityr/iro.hpp"

namespace ityr {

template <typename P>
class iro_context_if {
  using impl = typename P::template iro_context_impl_t<P>;
  using iro = typename P::iro;
  using access_mode = typename iro::access_mode;
  template <typename T>
  using global_ptr = typename iro::template global_ptr<T>;

public:
  template <access_mode Mode, typename T, typename Fn>
  static auto with_checkout(global_ptr<T> p, std::size_t n, Fn&& f) {
    return impl::template with_checkout<Mode>(p, n, std::forward<Fn>(f));
  }

  template <access_mode Mode1, access_mode Mode2,
            typename T1, typename T2, typename Fn>
  static auto with_checkout(global_ptr<T1> p1, std::size_t n1,
                            global_ptr<T2> p2, std::size_t n2,
                            Fn&& f) {
    return with_checkout<Mode1>(p1, n1, [&](auto&& p1_) {
      return with_checkout<Mode2>(p2, n2, [&](auto&& p2_) {
        return std::forward<Fn>(f)(std::forward<decltype(p1_)>(p1_),
                                   std::forward<decltype(p2_)>(p2_));
      });
    });
  }

  template <access_mode Mode1, access_mode Mode2, access_mode Mode3,
            typename T1, typename T2, typename T3, typename Fn>
  static auto with_checkout(global_ptr<T1> p1, std::size_t n1,
                            global_ptr<T2> p2, std::size_t n2,
                            global_ptr<T3> p3, std::size_t n3,
                            Fn&& f) {
    return with_checkout<Mode1>(p1, n1, [&](auto&& p1_) {
      return with_checkout<Mode2>(p2, n2, [&](auto&& p2_) {
        return with_checkout<Mode3>(p3, n3, [&](auto&& p3_) {
          return std::forward<Fn>(f)(std::forward<decltype(p1_)>(p1_),
                                     std::forward<decltype(p2_)>(p2_),
                                     std::forward<decltype(p3_)>(p3_));
        });
      });
    });
  }

  // It must be guaranteed that the thread is not migrated to another worker during execution of f
  template <access_mode Mode, typename T, typename Fn>
  static auto with_checkout_tied(global_ptr<T> p, std::size_t n, Fn&& f) {
    return impl::template with_checkout_tied<Mode>(p, n, std::forward<Fn>(f));
  }

  template <access_mode Mode1, access_mode Mode2,
            typename T1, typename T2, typename Fn>
  static auto with_checkout_tied(global_ptr<T1> p1, std::size_t n1,
                                 global_ptr<T2> p2, std::size_t n2,
                                 Fn&& f) {
    return with_checkout_tied<Mode1>(p1, n1, [&](auto&& p1_) {
      return with_checkout_tied<Mode2>(p2, n2, [&](auto&& p2_) {
        return std::forward<Fn>(f)(std::forward<decltype(p1_)>(p1_),
                                   std::forward<decltype(p2_)>(p2_));
      });
    });
  }

  template <access_mode Mode1, access_mode Mode2, access_mode Mode3,
            typename T1, typename T2, typename T3, typename Fn>
  static auto with_checkout_tied(global_ptr<T1> p1, std::size_t n1,
                                 global_ptr<T2> p2, std::size_t n2,
                                 global_ptr<T3> p3, std::size_t n3,
                                 Fn&& f) {
    return with_checkout_tied<Mode1>(p1, n1, [&](auto&& p1_) {
      return with_checkout_tied<Mode2>(p2, n2, [&](auto&& p2_) {
        return with_checkout_tied<Mode3>(p3, n3, [&](auto&& p3_) {
          return std::forward<Fn>(f)(std::forward<decltype(p1_)>(p1_),
                                     std::forward<decltype(p2_)>(p2_),
                                     std::forward<decltype(p3_)>(p3_));
        });
      });
    });
  }

  template <typename Fn, typename... Args>
  static auto with_checkout_cancel(Fn&& f, Args&&... args) {
    return impl::with_checkout_cancel(std::forward<Fn>(f), std::forward<Args>(args)...);
  }

};

template <typename P>
class iro_context_enabled {
  using iro = typename P::iro;
  using access_mode = typename iro::access_mode;
  template <typename T>
  using global_ptr = typename iro::template global_ptr<T>;

  struct checkout_entry {
    uintptr_t addr;
    std::size_t size;
    access_mode mode;
  };

  static std::vector<checkout_entry>& checkout_entries() {
    static std::vector<checkout_entry> entries;
    return entries;
  }

public:
  template <access_mode Mode, typename T, typename Fn>
  static auto with_checkout(global_ptr<T> p, std::size_t n, Fn&& f) {
    auto p_ = iro::template checkout<Mode>(p, n);
    auto& local_ces = checkout_entries();
    local_ces.push_back({reinterpret_cast<uintptr_t>(p_), n * sizeof(T), Mode});

    auto at_end = [&]() {
      local_ces.pop_back();
      iro::template checkin<Mode>(p_, n);
    };

    if constexpr (std::is_void_v<std::invoke_result_t<Fn, decltype(p_)>>) {
      std::forward<Fn>(f)(p_);
      at_end();
    } else {
      auto ret = std::forward<Fn>(f)(p_);
      at_end();
      return ret;
    }
  }

  template <access_mode Mode, typename T, typename Fn>
  static auto with_checkout_tied(global_ptr<T> p, std::size_t n, Fn&& f) {
    auto p_ = iro::template checkout<Mode>(p, n);

    if constexpr (std::is_void_v<std::invoke_result_t<Fn, decltype(p_)>>) {
      std::forward<Fn>(f)(p_);
      iro::template checkin<Mode>(p_, n);
    } else {
      auto ret = std::forward<Fn>(f)(p_);
      iro::template checkin<Mode>(p_, n);
      return ret;
    }
  }

  template <typename Fn, typename... Args>
  static auto with_checkout_cancel(Fn&& f, Args&&... args) {
    auto& local_ces = checkout_entries();
    if (!local_ces.empty()) {
      // FIXME: configurable maximum checkout entry num or flexible size
      constexpr int max_checkout_entries = 10;
      checkout_entry ces[max_checkout_entries];

      int count = 0;
      for (const auto& ce : local_ces) {
        if (ce.mode == access_mode::read) {
          iro::template checkin<access_mode::read>(reinterpret_cast<const std::byte*>(ce.addr), ce.size);
        } else {
          // assume that checkin behaves the same for read_write and read
          iro::template checkin<access_mode::write>(reinterpret_cast<std::byte*>(ce.addr), ce.size);
        }

        // save checkout entries in the stack region
        ces[count++] = ce;
        assert(count <= max_checkout_entries);
      }

      local_ces.clear();

      // called after the given function is finished
      auto at_end = [&]() {
        for (int i = 0; i < count; i++) {
          const auto& ce = ces[i];

          auto gptr = global_ptr<std::byte>(reinterpret_cast<std::byte*>(ce.addr));

          if (ce.mode == access_mode::read) {
            iro::template checkout<access_mode::read>(gptr, ce.size);
          } else {
            // Change the access mode from write to read_write so that
            // the changes by the previous thread are visible to successors
            iro::template checkout<access_mode::read_write>(gptr, ce.size);
          }

          // restore checkout entries
          local_ces.push_back(ce);
        }
      };

      if constexpr (std::is_void_v<std::invoke_result_t<Fn, Args...>>) {
        std::forward<Fn>(f)(std::forward<Args>(args)...);
        at_end();

      } else {
        auto ret = std::forward<Fn>(f)(std::forward<Args>(args)...);
        at_end();
        return ret;
      }

    } else {
      return std::forward<Fn>(f)(std::forward<Args>(args)...);
    }
  }

};

// no management for checkout context
template <typename P>
class iro_context_disabled {
  using iro = typename P::iro;
  using access_mode = typename iro::access_mode;
  template <typename T>
  using global_ptr = typename iro::template global_ptr<T>;

public:
  template <access_mode Mode, typename T, typename Fn>
  static auto with_checkout(global_ptr<T> p, std::size_t n, Fn&& f) {
    auto p_ = iro::template checkout<Mode>(p, n);

    if constexpr (std::is_void_v<std::invoke_result_t<Fn, decltype(p_)>>) {
      std::forward<Fn>(f)(p_);
      iro::template checkin<Mode>(p_, n);
    } else {
      auto ret = std::forward<Fn>(f)(p_);
      iro::template checkin<Mode>(p_, n);
      return ret;
    }
  }

  template <access_mode Mode, typename T, typename Fn>
  static auto with_checkout_tied(global_ptr<T> p, std::size_t n, Fn&& f) {
    return with_checkout<Mode>(p, n, std::forward<Fn>(f));
  }

  template <typename Fn, typename... Args>
  static auto with_checkout_cancel(Fn&& f, Args&&... args) {
    return std::forward<Fn>(f)(std::forward<Args>(args)...);
  }
};

struct iro_context_policy_default {
  template <typename P>
  using iro_context_impl_t = iro_context_enabled<P>;
  using iro = iro_if<iro_policy_default>;
  static int rank() { return 0; }
  static int n_ranks() { return 1; }
};

}
