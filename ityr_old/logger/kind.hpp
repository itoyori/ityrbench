#pragma once

#include <cstdlib>

namespace ityr {
namespace logger {

template <typename Derived, typename Value>
class kind_base {
public:
  using value = Value;

  constexpr kind_base(Value val) : val_(val) {}

  constexpr bool operator==(Derived k) const { return val_ == k.val_; }
  constexpr bool operator!=(Derived k) const { return val_ != k.val_; }

  constexpr bool included(Derived kinds[], int n) const {
    return n > 0 && (*this == kinds[0] || included(kinds + 1, n - 1));
  }

#ifndef ITYR_LOGGER_ENABLED_KINDS
#define ITYR_LOGGER_ENABLED_KINDS
#endif

#ifndef ITYR_LOGGER_DISABLED_KINDS
#define ITYR_LOGGER_DISABLED_KINDS
#endif

  constexpr bool is_valid() const {
    Derived enabled_kinds[]  = {Value::_NKinds, ITYR_LOGGER_ENABLED_KINDS};
    Derived disabled_kinds[] = {Value::_NKinds, ITYR_LOGGER_DISABLED_KINDS};

    constexpr int n_enabled = sizeof(enabled_kinds) / sizeof(Derived);
    constexpr int n_disabled = sizeof(disabled_kinds) / sizeof(Derived);
    static_assert(!(n_enabled > 1 && n_disabled > 1),
                  "Enabled kinds and disabled kinds cannot be specified at the same time.");

    if (n_enabled > 1) {
      return included(enabled_kinds + 1, n_enabled - 1);
    } else if (n_disabled > 1) {
      return !included(disabled_kinds + 1, n_disabled - 1);
    } else {
      return true;
    }
  }

#undef ITYR_LOGGER_ENABLED_KINDS
#undef ITYR_LOGGER_DISABLED_KINDS

  static constexpr std::size_t size() {
    return (std::size_t)Value::_NKinds;
  }

  constexpr std::size_t index() const {
    return (std::size_t)val_;
  }

protected:
  const Value val_;
};

enum class kind_dummy_value { _NKinds = 0 };

class kind_dummy : public kind_base<kind_dummy, kind_dummy_value> {
public:
  using ityr::logger::kind_base<kind_dummy, kind_dummy_value>::kind_base;
  constexpr const char* str() const { return ""; }
};

}
}
