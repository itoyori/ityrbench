#pragma once

#include <cstdlib>
#include <cassert>

#include "pcas/pcas.hpp"

#include "ityr/util.hpp"
#include "ityr/iro.hpp"
#include "ityr/iro_context.hpp"
#include "ityr/ito_pattern.hpp"

namespace ityr {

// TODO: remove it and use std::span in C++20
template <typename T>
class raw_span {
  using this_t = raw_span<T>;

public:
  using element_type = T;
  using value_type   = std::remove_cv_t<T>;
  using size_type    = std::size_t;
  using pointer      = T*;
  using iterator     = pointer;
  using reference    = T&;

private:
  pointer ptr_ = nullptr;
  size_type n_ = 0;

public:
  raw_span() {}
  template <typename ContiguousIterator>
  raw_span(ContiguousIterator first, size_type n) :
      ptr_(&(*first)), n_(n) {}
  template <typename ContiguousIterator>
  raw_span(ContiguousIterator first, ContiguousIterator last) :
      ptr_(&(*first)), n_(last - first) {}
  template <typename U>
  raw_span(raw_span<U> s) : ptr_(s.data()), n_(s.size() * sizeof(U) / sizeof(T)) {}

  constexpr pointer data() const noexcept { return ptr_; }
  constexpr size_type size() const noexcept { return n_; }

  constexpr iterator begin() const noexcept { return ptr_; }
  constexpr iterator end() const noexcept { return ptr_ + n_; }

  constexpr reference operator[](size_type i) const { assert(i <= n_); return ptr_[i]; }

  constexpr reference front() const { return *ptr_; }
  constexpr reference back() const { return *(ptr_ + n_ - 1); }

  constexpr bool empty() const noexcept { return n_ == 0; }

  constexpr this_t subspan(size_type offset, size_type count) const {
    assert(offset + count <= n_);
    return {ptr_ + offset, count};
  }
};

template <typename T>
inline constexpr auto data(const raw_span<T>& s) noexcept {
  return s.data();
}

template <typename T>
inline constexpr auto size(const raw_span<T>& s) noexcept {
  return s.size();
}

template <typename T>
inline constexpr auto begin(const raw_span<T>& s) noexcept {
  return s.begin();
}

template <typename T>
inline constexpr auto end(const raw_span<T>& s) noexcept {
  return s.end();
}

template <pcas::access_mode Mode1,
          typename T1, typename Fn>
inline auto with_checkout(const raw_span<T1>& s,
                          Fn f) {
  return f(s);
}

template <pcas::access_mode Mode1,
          pcas::access_mode Mode2,
          typename T1, typename T2, typename Fn>
inline auto with_checkout(const raw_span<T1>& s1,
                          const raw_span<T2>& s2,
                          Fn f) {
  return f(s1, s2);
}

template <pcas::access_mode Mode1,
          pcas::access_mode Mode2,
          pcas::access_mode Mode3,
          typename T1, typename T2, typename T3, typename Fn>
inline auto with_checkout(const raw_span<T1>& s1,
                          const raw_span<T2>& s2,
                          const raw_span<T3>& s3,
                          Fn f) {
  return f(s1, s2, s3);
}

template <pcas::access_mode Mode1,
          typename T1, typename Fn>
inline auto with_checkout_tied(const raw_span<T1>& s,
                               Fn f) {
  return f(s);
}

template <pcas::access_mode Mode1,
          pcas::access_mode Mode2,
          typename T1, typename T2, typename Fn>
inline auto with_checkout_tied(const raw_span<T1>& s1,
                               const raw_span<T2>& s2,
                               Fn f) {
  return f(s1, s2);
}

template <pcas::access_mode Mode1,
          pcas::access_mode Mode2,
          pcas::access_mode Mode3,
          typename T1, typename T2, typename T3, typename Fn>
inline auto with_checkout_tied(const raw_span<T1>& s1,
                               const raw_span<T2>& s2,
                               const raw_span<T3>& s3,
                               Fn f) {
  return f(s1, s2, s3);
}

struct global_vector_options {
  bool collective = false;
  bool parallel_construct = false;
  bool parallel_destruct = false;
  std::size_t cutoff = 1024;
};

template <typename P>
struct global_container_if {
  using iro = typename P::iro;
  using iro_context = typename P::iro_context;
  using ito_pattern = typename P::ito_pattern;
  using access_mode = typename iro::access_mode;
  template <typename T>
  using global_ptr = typename iro::template global_ptr<T>;

  template <typename T>
  class global_span {
    using this_t = global_span<T>;

  public:
    using element_type = T;
    using value_type   = std::remove_cv_t<T>;
    using size_type    = std::size_t;
    using pointer      = global_ptr<T>;
    using iterator     = pointer;
    using reference    = typename std::iterator_traits<pointer>::reference;

    using policy = P;

  private:
    pointer ptr_ = nullptr;
    size_type n_ = 0;

  public:
    global_span() {}
    template <typename ContiguousIterator>
    global_span(ContiguousIterator first, size_type n) :
        ptr_(&(*first)), n_(n) {}
    template <typename ContiguousIterator>
    global_span(ContiguousIterator first, ContiguousIterator last) :
        ptr_(&(*first)), n_(last - first) {}
    template <typename U>
    global_span(global_span<U> s) : ptr_(s.data()), n_(s.size() * sizeof(U) / sizeof(T)) {}

    pointer data() const noexcept { return ptr_; }
    size_type size() const noexcept { return n_; }

    pointer begin() const noexcept { return ptr_; }
    pointer end() const noexcept { return ptr_ + n_; }

    reference operator[](size_type i) const { assert(i < n_); return ptr_[i]; }

    reference front() const { return *ptr_; }
    reference back() const { return *(ptr_ + n_ - 1); }

    bool empty() const noexcept { return n_ == 0; }

    this_t subspan(size_type offset, size_type count) const {
      assert(offset + count <= n_);
      return {ptr_ + offset, count};
    }

    friend auto data(const raw_span<T>& s) noexcept {
      return s.data();
    }

    friend auto size(const raw_span<T>& s) noexcept {
      return s.size();
    }

    friend auto begin(const raw_span<T>& s) noexcept {
      return s.begin();
    }

    friend auto end(const raw_span<T>& s) noexcept {
      return s.end();
    }

  };

  template <typename T>
  class global_vector {
    using this_t = global_vector<T>;

  public:
    using value_type      = T;
    using size_type       = std::size_t;
    using pointer         = global_ptr<T>;
    using const_pointer   = global_ptr<std::add_const_t<T>>;
    using iterator        = pointer;
    using const_iterator  = const_pointer;
    using difference_type = typename std::iterator_traits<pointer>::difference_type;
    using reference       = typename std::iterator_traits<pointer>::reference;
    using const_reference = typename std::iterator_traits<const_pointer>::reference;

    using policy = P;

  private:
    pointer begin_        = nullptr;
    pointer end_          = nullptr;
    pointer reserved_end_ = nullptr;

    global_vector_options opts_;

    size_type next_size(size_type least) const {
      return std::max(least, size() * 2);
    }

    pointer allocate_mem(size_type count) const {
      if (opts_.collective) {
        return iro::template malloc<T>(count);
      } else {
        return iro::template malloc_local<T>(count);
      }
    }

    void free_mem(pointer p, size_type count) const {
      iro::template free<T>(p, count);
    }

    template <typename Fn, typename... Args>
    auto master_do_if_coll(Fn&& f, Args&&... args) const {
      if (opts_.collective) {
        return ito_pattern::master_do(std::forward<Fn>(f), std::forward<Args>(args)...);
      } else {
        return std::forward<Fn>(f)(std::forward<Args>(args)...);
      }
    }

    template <typename... Args>
    void initialize_uniform(size_type count, Args&&... args) {
      begin_ = allocate_mem(count);
      end_ = begin_ + count;
      reserved_end_ = begin_ + count;

      construct_elems(begin(), end(), std::forward<Args>(args)...);
    }

    template <typename InputIterator>
    void initialize_from_iter(InputIterator first, InputIterator last, std::input_iterator_tag) {
      assert(!opts_.collective);
      assert(!opts_.parallel_construct);

      for (; first != last; ++first) {
        emplace_back(*first);
      }
    }

    template <typename ForwardIterator>
    void initialize_from_iter(ForwardIterator first, ForwardIterator last, std::forward_iterator_tag) {
      auto d = std::distance(first, last);

      if (d > 0) {
        begin_ = allocate_mem(d);
        end_ = begin_ + d;
        reserved_end_ = begin_ + d;

        construct_elems_from_iter(first, last, begin());

      } else {
        begin_ = end_ = reserved_end_ = nullptr;
      }
    }

    template <typename... Args>
    void construct_elems(pointer b, pointer e, Args&&... args) const {
      master_do_if_coll([=]() {
        if (opts_.parallel_construct) {
          ito_pattern::template parallel_for<access_mode::write>(
              b, e, [=](auto&& x) { new (&x) T(args...); }, opts_.cutoff);
        } else {
          ito_pattern::template serial_for<access_mode::write>(
              b, e, [&](auto&& x) { new (&x) T(std::forward<Args>(args)...); }, opts_.cutoff);
        }
      });
    }

    template <typename ForwardIterator>
    void construct_elems_from_iter(ForwardIterator first, ForwardIterator last, pointer b) const {
      master_do_if_coll([=]() {
        if constexpr (is_const_iterator_v<ForwardIterator>) {
          if (opts_.parallel_construct) {
            ito_pattern::template parallel_for<access_mode::read, access_mode::write>(
                first, last, b, [](const auto& src, auto&& x) { new (&x) T(src); }, opts_.cutoff);
          } else {
            ito_pattern::template serial_for<access_mode::read, access_mode::write>(
                first, last, b, [](const auto& src, auto&& x) { new (&x) T(src); }, opts_.cutoff);
          }
        } else {
          if (opts_.parallel_construct) {
            ito_pattern::template parallel_for<access_mode::read_write, access_mode::write>(
                first, last, b, [](auto&& src, auto&& x) { new (&x) T(std::forward<decltype(src)>(src)); }, opts_.cutoff);
          } else {
            ito_pattern::template serial_for<access_mode::read_write, access_mode::write>(
                first, last, b, [](auto&& src, auto&& x) { new (&x) T(std::forward<decltype(src)>(src)); }, opts_.cutoff);
          }
        }
      });
    }

    void destruct_elems(pointer b, pointer e) const {
      if constexpr (!std::is_trivially_destructible_v<T>) {
        master_do_if_coll([=]() {
          if (opts_.parallel_destruct) {
            ito_pattern::template parallel_for<access_mode::read_write>(
                b, e, [](auto&& x) { std::destroy_at(&x); }, opts_.cutoff);
          } else {
            ito_pattern::template serial_for<access_mode::read_write>(
                b, e, [](auto&& x) { std::destroy_at(&x); }, opts_.cutoff);
          }
        });
      }
    }

    void realloc_mem(size_type count) {
      pointer old_begin = begin_;
      pointer old_end = end_;
      size_type old_capacity = capacity();

      begin_ = allocate_mem(count);
      end_ = begin_ + (old_end - old_begin);
      reserved_end_ = begin_ + count;

      if (old_end - old_begin > 0) {
        construct_elems_from_iter(ityr::make_move_iterator(old_begin),
                                  ityr::make_move_iterator(old_end),
                                  begin());

        destruct_elems(old_begin, old_end);
      }

      if (old_capacity > 0) {
        free_mem(old_begin, old_capacity);
      }
    }

    template <typename... Args>
    void resize_impl(size_type count, Args&&... args) {
      if (count > size()) {
        if (count > capacity()) {
          size_type new_cap = next_size(count);
          realloc_mem(new_cap);
        }
        construct_elems(end(), begin() + count, std::forward<Args>(args)...);
        end_ = begin() + count;

      } else if (count < size()) {
        destruct_elems(begin() + count, end());
        end_ = begin() + count;
      }
    }

    template <typename... Args>
    void push_back_impl(Args&&... args) {
      assert(!opts_.collective);
      if (size() == capacity()) {
        size_type new_cap = next_size(size() + 1);
        realloc_mem(new_cap);
      }
      iro_context::template with_checkout_tied<access_mode::write>(end(), 1,
          [&](auto&& p) { new (p) T(std::forward<Args>(args)...); });
      ++end_;
    }

  public:
    global_vector() : global_vector(global_vector_options()) {}
    explicit global_vector(size_type count) : global_vector(global_vector_options(), count) {}
    explicit global_vector(size_type count, const T& value) : global_vector(global_vector_options(), count, value) {}
    template <typename InputIterator>
    global_vector(InputIterator first, InputIterator last) : global_vector(global_vector_options(), first, last) {}

    explicit global_vector(const global_vector_options& opts) : opts_(opts) {}

    explicit global_vector(const global_vector_options& opts, size_type count) : opts_(opts) {
      initialize_uniform(count);
    }

    explicit global_vector(const global_vector_options& opts, size_type count, const T& value) : opts_(opts) {
      initialize_uniform(count, value);
    }

    template <typename InputIterator>
    global_vector(const global_vector_options& opts, InputIterator first, InputIterator last) : opts_(opts) {
      initialize_from_iter(first, last,
                           typename std::iterator_traits<InputIterator>::iterator_category());
    }

    ~global_vector() {
      if (begin() != nullptr) {
        destruct_elems(begin(), end());
        free_mem(begin(), capacity());
      }
    }

    global_vector(const this_t& other) : opts_(other.options()) {
      initialize_from_iter(other.cbegin(), other.cend(), std::random_access_iterator_tag{});
    }
    this_t& operator=(const this_t& other) {
      // TODO: skip freeing memory and reuse it when it has enough amount of memory
      this->~global_vector();
      // should we copy options?
      opts_ = other.options();
      initialize_from_iter(other.cbegin(), other.cend(), std::random_access_iterator_tag{});
      return *this;
    }

    global_vector(this_t&& other) :
        begin_(other.begin_),
        end_(other.end_),
        reserved_end_(other.reserved_end_),
        opts_(other.opts_) {
      other.begin_ = other.end_ = other.reserved_end_ = nullptr;
    }
    this_t& operator=(this_t&& other) {
      this->~global_vector();
      begin_ = other.begin_;
      end_ = other.end_;
      reserved_end_ = other.reserved_end_;
      opts_ = other.opts_;
      other.begin_ = other.end_ = other.reserved_end_ = nullptr;
      return *this;
    }

    pointer data() const noexcept { return begin_; }
    size_type size() const noexcept { return end_ - begin_; }
    size_type capacity() const noexcept { return reserved_end_ - begin_; }

    global_vector_options options() const noexcept { return opts_; }

    iterator begin() const noexcept { return begin_; }
    iterator end() const noexcept { return end_; }

    const_iterator cbegin() const noexcept { return const_iterator(begin_); }
    const_iterator cend() const noexcept { return const_iterator(end_); }

    reference operator[](size_type i) const {
      assert(i <= size());
      return *(begin() + i);
    }
    reference at(size_type i) const {
      if (i >= size()) {
        std::stringstream ss;
        ss << "Global vector: Index " << i << " is out of range [0, " << size() << ").";
        throw std::out_of_range(ss.str());
      }
      return (*this)[i];
    }

    reference front() const { return *begin(); }
    reference back() const { return *(end() - 1); }

    bool empty() const noexcept { return size() == 0; }

    void swap(this_t& other) noexcept {
      using std::swap;
      swap(begin_, other.begin_);
      swap(end_, other.end_);
      swap(reserved_end_, other.reserved_end_);
      swap(opts_, other.opts_);
    }

    void clear() {
      destruct_elems();
      end_ = begin();
    }

    void reserve(size_type new_cap) {
      if (capacity() == 0 && new_cap > 0) {
        begin_ = allocate_mem(new_cap);
        end_ = begin_;
        reserved_end_ = begin_ + new_cap;

      } else if (new_cap > capacity()) {
        realloc_mem(new_cap);
      }
    }

    void resize(size_type count) {
      resize_impl(count);
    }

    void resize(size_type count, const value_type& value) {
      resize_impl(count, value);
    }

    void push_back(const value_type& value) {
      push_back_impl(value);
    }

    void push_back(value_type&& value) {
      push_back_impl(std::move(value));
    }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
      push_back_impl(std::forward<Args>(args)...);
      return back();
    }

    void pop_back() {
      assert(!opts_.collective);
      assert(size() > 0);
      iro_context::template with_checkout_tied<access_mode::read_write>(end() - 1, 1,
          [&](auto&& x) { std::destroy_at(&x); });
      --end_;
    }

    friend void swap(this_t& v1, this_t& v2) noexcept {
      v1.swap(v2);
    }

  };

};

// TODO: we would like to move these with_checkout calls to the inner class
// and make them friend, but we cannot do it because functions with explicit
// template parameters (Modes in our case) are not candidates for ADL in C++17.
// (this issue is resolved in C++20).
template <pcas::access_mode Mode,
          typename GlobalSpan, typename Fn>
inline auto with_checkout(GlobalSpan s, Fn f) {
  using T = typename GlobalSpan::element_type;
  using iro_context = typename GlobalSpan::policy::iro_context;
  return iro_context::template with_checkout<Mode>(s.data(), s.size(),
                                                   [&](auto&& p) {
    return f(raw_span<T>{p, s.size()});
  });
}

template <pcas::access_mode Mode1,
          pcas::access_mode Mode2,
          typename GlobalSpan1, typename GlobalSpan2, typename Fn>
inline auto with_checkout(GlobalSpan1 s1, GlobalSpan2 s2, Fn f) {
  using T1 = typename GlobalSpan1::element_type;
  using T2 = typename GlobalSpan2::element_type;
  using iro_context = typename GlobalSpan1::policy::iro_context;
  return iro_context::template with_checkout<Mode1, Mode2>(s1.data(), s1.size(),
                                                           s2.data(), s2.size(),
                                                           [&](auto&& p1, auto&& p2) {
    return f(raw_span<T1>{p1, s1.size()}, raw_span<T2>{p2, s2.size()});
  });
}

template <pcas::access_mode Mode1,
          pcas::access_mode Mode2,
          pcas::access_mode Mode3,
          typename GlobalSpan1, typename GlobalSpan2, typename GlobalSpan3, typename Fn>
inline auto with_checkout(GlobalSpan1 s1, GlobalSpan2 s2, GlobalSpan3 s3, Fn f) {
  using T1 = typename GlobalSpan1::element_type;
  using T2 = typename GlobalSpan2::element_type;
  using T3 = typename GlobalSpan3::element_type;
  using iro_context = typename GlobalSpan1::policy::iro_context;
  return iro_context::template with_checkout<Mode1, Mode2, Mode3>(s1.data(), s1.size(),
                                                                  s2.data(), s2.size(),
                                                                  s3.data(), s3.size(),
                                                                  [&](auto&& p1, auto&& p2, auto&& p3) {
    return f(raw_span<T1>{p1, s1.size()}, raw_span<T2>{p2, s2.size()}, raw_span<T3>{p3, s3.size()});
  });
}

template <pcas::access_mode Mode,
          typename GlobalSpan, typename Fn>
inline auto with_checkout_tied(GlobalSpan s, Fn f) {
  using T = typename GlobalSpan::element_type;
  using iro_context = typename GlobalSpan::policy::iro_context;
  return iro_context::template with_checkout_tied<Mode>(s.data(), s.size(),
                                                        [&](auto&& p) {
    return f(raw_span<T>{p, s.size()});
  });
}

template <pcas::access_mode Mode1,
          pcas::access_mode Mode2,
          typename GlobalSpan1, typename GlobalSpan2, typename Fn>
inline auto with_checkout_tied(GlobalSpan1 s1, GlobalSpan2 s2, Fn f) {
  using T1 = typename GlobalSpan1::element_type;
  using T2 = typename GlobalSpan2::element_type;
  using iro_context = typename GlobalSpan1::policy::iro_context;
  return iro_context::template with_checkout_tied<Mode1, Mode2>(s1.data(), s1.size(),
                                                                s2.data(), s2.size(),
                                                                [&](auto&& p1, auto&& p2) {
    return f(raw_span<T1>{p1, s1.size()}, raw_span<T2>{p2, s2.size()});
  });
}

template <pcas::access_mode Mode1,
          pcas::access_mode Mode2,
          pcas::access_mode Mode3,
          typename GlobalSpan1, typename GlobalSpan2, typename GlobalSpan3, typename Fn>
inline auto with_checkout_tied(GlobalSpan1 s1, GlobalSpan2 s2, GlobalSpan3 s3, Fn f) {
  using T1 = typename GlobalSpan1::element_type;
  using T2 = typename GlobalSpan2::element_type;
  using T3 = typename GlobalSpan3::element_type;
  using iro_context = typename GlobalSpan1::policy::iro_context;
  return iro_context::template with_checkout_tied<Mode1, Mode2, Mode3>(s1.data(), s1.size(),
                                                                       s2.data(), s2.size(),
                                                                       s3.data(), s3.size(),
                                                                       [&](auto&& p1, auto&& p2, auto&& p3) {
    return f(raw_span<T1>{p1, s1.size()}, raw_span<T2>{p2, s2.size()}, raw_span<T3>{p3, s3.size()});
  });
}

struct global_container_policy_default {
  using iro = iro_if<iro_policy_default>;
  using iro_context = iro_context_if<iro_context_policy_default>;
  using ito_pattern = ito_pattern_if<ito_pattern_policy_default>;
};

}
