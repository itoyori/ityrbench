#pragma once

#include <csignal>
#include <dlfcn.h>

#include "ityr/ityr.hpp"
#include "pcg_random.hpp"
#include "backward.hpp"

template <typename T> constexpr inline const char* typename_str()         { return "unknown"; }
template <>           constexpr inline const char* typename_str<char>()   { return "char";    }
template <>           constexpr inline const char* typename_str<short>()  { return "short";   }
template <>           constexpr inline const char* typename_str<int>()    { return "int";     }
template <>           constexpr inline const char* typename_str<long>()   { return "long";    }
template <>           constexpr inline const char* typename_str<float>()  { return "float";   }
template <>           constexpr inline const char* typename_str<double>() { return "double";  }

template <typename T>
constexpr inline bool is_pow2(T x) {
  return !(x & (x - 1));
}

template <typename T>
constexpr inline T round_down_pow2(T x, T alignment) {
  return x & ~(alignment - 1);
}

template <typename T>
constexpr inline T round_up_pow2(T x, T alignment) {
  return (x + alignment - 1) & ~(alignment - 1);
}

template <typename T>
constexpr inline auto log2_pow2(T n) {
  int x = 0;
  while (n >>= 1) x++;
  return x;
}

inline timer_t* abort_timer_instance() {
  static timer_t timer;
  return &timer;
}

inline void init_abort_timer() {
  struct sigevent sev;
  sev.sigev_notify = SIGEV_SIGNAL;
  sev.sigev_signo = SIGABRT;
  if (timer_create(CLOCK_REALTIME, &sev, abort_timer_instance()) == -1) {
    perror("timer_create");
    exit(1);
  }
}

inline void set_abort_timeout(int timeout_sec) {
  struct itimerspec its;
  its.it_value.tv_sec = timeout_sec;
  its.it_value.tv_nsec = 0;
  its.it_interval.tv_sec = 0;
  its.it_interval.tv_nsec = 0;

  timer_settime(*abort_timer_instance(), 0, &its, NULL);
}

inline void print_backtrace(void* ctx = nullptr) {
  backward::StackTrace st;
  st.load_here(32, ctx);
  backward::Printer p;
  p.object = true;
  p.color_mode = backward::ColorMode::always;
  p.address = true;
  p.reverse = false;
  p.print(st, stderr);
  fflush(stderr);
}

inline void signal_handler(int sig, siginfo_t* si, void* ctx) {
  // cancel signal handler for SIGABRT
  signal(SIGABRT, SIG_DFL);

  /* sleep(1000000); */

  // Set timeout so as not to get stuck in the signal handler.
  // Non-async-signal-safe functions (e.g., printf) are called in the following,
  // which can result in a deadlock.
  set_abort_timeout(10);

  auto t = ityr::gettime_ns();

  if (sig == 11) { // SIGSEGV
    fprintf(stderr, "%ld: Signal %d received (at %p).\n", t, sig, si->si_addr);
  } else {
    fprintf(stderr, "%ld: Signal %d received.\n", t, sig);
  }
  fflush(stderr);

  print_backtrace(ctx);

  std::abort();
}

inline void set_signal_handler(int sig) {
  struct sigaction sa;
  sa.sa_flags     = SA_SIGINFO;
  sa.sa_sigaction = signal_handler;
  sigemptyset(&sa.sa_mask);
  if (sigaction(sig, &sa, NULL) == -1) {
    perror("sigacton");
    exit(1);
  }
}

inline void set_signal_handlers() {
  init_abort_timer();

  set_signal_handler(SIGSEGV);
  set_signal_handler(SIGABRT);
  set_signal_handler(SIGBUS);
  set_signal_handler(SIGTERM);
}
