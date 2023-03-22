#pragma once

#include <cstdio>
#include <cstdint>

#include <mpi.h>

#include "ityr/util.hpp"

namespace ityr {
namespace logger {

template <typename P>
class impl_stats {
public:
  using begin_data_t = uint64_t;

private:
  using this_t = impl_stats;
  using kind = typename P::logger_kind_t;
  using wallclock = typename P::wallclock_t;

  int rank_;
  int n_ranks_;

  uint64_t t_begin_;
  uint64_t t_end_;

  bool     stat_print_per_rank_;
  uint64_t stat_acc_[kind::size()];
  uint64_t stat_acc_total_[kind::size()];
  uint64_t stat_count_[kind::size()];
  uint64_t stat_count_total_[kind::size()];

  static this_t& get_instance_() {
    static this_t my_instance;
    return my_instance;
  }

  static void print_kind_stat_(kind k, int rank) {
    if (k.is_valid()) {
      this_t& lgr = get_instance_();
      if (lgr.stat_print_per_rank_) {
        uint64_t acc = lgr.stat_acc_[k.index()];
        uint64_t acc_total = lgr.t_end_ - lgr.t_begin_;
        uint64_t count = lgr.stat_count_[k.index()];
        printf("(Rank %3d) %-23s : %10.6f %% ( %15ld ns / %15ld ns ) count: %8ld ave: %8ld ns\n",
               rank, k.str(), (double)acc / acc_total * 100, acc, acc_total, count, count == 0 ? 0 : (acc / count));
      } else {
        uint64_t acc = lgr.stat_acc_total_[k.index()];
        uint64_t acc_total = (lgr.t_end_ - lgr.t_begin_) * lgr.n_ranks_;
        uint64_t count = lgr.stat_count_total_[k.index()];
        printf("  %-23s : %10.6f %% ( %15ld ns / %15ld ns ) count: %8ld ave: %8ld ns\n",
               k.str(), (double)acc / acc_total * 100, acc, acc_total, count, count == 0 ? 0 : (acc / count));
      }
    }
  }

  template <typename kind::value K>
  static void acc_stat_(uint64_t t0, uint64_t t1) {
    this_t& lgr = get_instance_();
    lgr.stat_acc_[kind(K).index()] += t1 - t0;
    lgr.stat_count_[kind(K).index()]++;
  }

  static void acc_init_() {
    this_t& lgr = get_instance_();
    for (size_t k = 0; k < kind::size(); k++) {
      lgr.stat_acc_[k] = 0;
      lgr.stat_acc_total_[k] = 0;
      lgr.stat_count_[k] = 0;
      lgr.stat_count_total_[k] = 0;
    }
  }

  static void print_stat_(int rank) {
    for (size_t k = 1; k < kind::size(); k++) {
      print_kind_stat_(kind((typename kind::value)k), rank);
    }
    printf("\n");
  }

public:
  static void init(int rank, int n_ranks) {
    this_t& lgr = get_instance_();
    lgr.rank_ = rank;
    lgr.n_ranks_ = n_ranks;

    lgr.stat_print_per_rank_ = get_env("MADM_LOGGER_PRINT_STAT_PER_RANK", false, rank);

    acc_init_();
  }

  static void flush(uint64_t t_begin, uint64_t t_end) {}

  static void flush_and_print_stat(uint64_t t_begin, uint64_t t_end) {
    this_t& lgr = get_instance_();

    // TODO: it is not easy to accurately show only events within [t_begin, t_end]
    lgr.t_begin_ = t_begin;
    lgr.t_end_ = t_end;

    if (lgr.stat_print_per_rank_) {
      if (lgr.rank_ == 0) {
        print_stat_(0);
        for (int i = 1; i < lgr.n_ranks_; i++) {
          MPI_Recv(lgr.stat_acc_, kind::size(), MPI_UINT64_T,
                   i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(lgr.stat_count_, kind::size(), MPI_UINT64_T,
                   i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          print_stat_(i);
        }
      } else {
        MPI_Send(lgr.stat_acc_, kind::size(), MPI_UINT64_T,
                 0, 0, MPI_COMM_WORLD);
        MPI_Send(lgr.stat_count_, kind::size(), MPI_UINT64_T,
                 0, 0, MPI_COMM_WORLD);
      }
    } else {
      MPI_Reduce(lgr.stat_acc_, lgr.stat_acc_total_, kind::size(),
                 MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(lgr.stat_count_, lgr.stat_count_total_, kind::size(),
                 MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
      if (lgr.rank_ == 0) {
        print_stat_(0);
      }
    }
    fflush(stdout);

    acc_init_();
  }

  static void warmup() {}

  static void clear() {
    acc_init_();
  }

  template <typename kind::value K>
  static begin_data_t begin_event() {
    if (kind(K).is_valid()) {
      return wallclock::get_time();
    } else {
      return 0;
    }
  }

  template <typename kind::value K>
  static void end_event(begin_data_t bd) {
    if (kind(K).is_valid()) {
      uint64_t t = wallclock::get_time();
      acc_stat_<K>(bd, t);
    }
  }

  template <typename kind::value K, typename Misc>
  static void end_event(begin_data_t bd, Misc m) {
    end_event<K>(bd);
  }
};

}
}
