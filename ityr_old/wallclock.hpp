#pragma once

#include <cstdint>
#include <unistd.h>

#include <madm_global_clock.h>

namespace ityr {

class wallclock_native {
public:
  static void init() {}
  static void sync() {}
  static uint64_t get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + (uint64_t)ts.tv_nsec;
  }
};

class wallclock_madm {
public:
  static void init() {
    madi::global_clock::init();
  }
  static void sync() {
    madi::global_clock::sync();
  }
  static uint64_t get_time() {
    return madi::global_clock::get_time();
  }
};

}
