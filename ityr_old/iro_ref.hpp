#pragma once

#include "pcas/pcas.hpp"

namespace ityr {

template <typename P, typename GPtrT>
class iro_ref : public pcas::global_ref_base<GPtrT> {
  using base_t = pcas::global_ref_base<GPtrT>;
  using this_t = iro_ref;
  using ptr_t = GPtrT;
  using value_t = typename GPtrT::value_type;
  using iro = typename P::iro;

  using base_t::ptr_;

  template <typename Fn>
  void with_read_write(Fn&& f) {
    value_t* vp = iro::template checkout<iro::access_mode::read_write>(ptr_, 1);
    std::forward<Fn>(f)(*vp);
    iro::template checkin<iro::access_mode::read_write>(vp, 1);
  }

public:
  using base_t::base_t;

  iro_ref(const this_t&) = default;
  iro_ref(this_t&&) = default;

  operator value_t() const {
    std::remove_const_t<value_t> ret;
    iro::get(ptr_, &ret, 1);
    return ret;
  }

  this_t& operator=(const value_t& v) {
    with_read_write([&](value_t& this_v) { this_v = v; });
    return *this;
  }

  this_t& operator=(value_t&& v) {
    with_read_write([&](value_t& this_v) { this_v = std::move(v); });
    return *this;
  }

  this_t& operator=(const this_t& r) {
    value_t v = r;
    return (*this = v);
  }

  this_t& operator=(this_t&& r) {
    return (*this = value_t(r));
  }

  this_t& operator+=(const value_t& v) {
    with_read_write([&](value_t& this_v) { this_v += v; });
    return *this;
  }

  this_t& operator-=(const value_t& v) {
    with_read_write([&](value_t& this_v) { this_v -= v; });
    return *this;
  }

  this_t& operator++() {
    with_read_write([&](value_t& this_v) { ++this_v; });
    return *this;
  }

  this_t& operator--() {
    with_read_write([&](value_t& this_v) { --this_v; });
    return *this;
  }

  value_t operator++(int) {
    value_t ret;
    with_read_write([&](value_t& this_v) { ret = this_v++; });
    return ret;
  }

  value_t operator--(int) {
    value_t ret;
    with_read_write([&](value_t& this_v) { ret = this_v--; });
    return ret;
  }

  void swap(this_t r) {
    PCAS_CHECK(&r != ptr_);
    with_read_write([&](value_t& this_v) {
      r.with_read_write([&](value_t& v) {
        using std::swap;
        swap(this_v, v);
      });
    });
  }
};

template <typename P, typename GPtrT>
inline void swap(iro_ref<P, GPtrT> r1, iro_ref<P, GPtrT> r2) {
  r1.swap(r2);
}

}
