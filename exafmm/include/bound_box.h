#ifndef bound_box_h
#define bound_box_h
#include "logger.h"
#include "namespace.h"
#include "types.h"

namespace EXAFMM_NAMESPACE {
  class BoundBox {
  public:
    //! Get Xmin and Xmax of bodies
    Bounds getBounds(Bodies & bodies) {
      logger::startTimer("Get bounds");                         // Start timer
      Bounds bounds;                                            // Bounds : Contains Xmin, Xmax
      if (bodies.empty()) {                                     // If body vector is empty
	bounds.Xmin = bounds.Xmax = 0;                          //  Set bounds to 0
      } else {                                                  // If body vector is not empty
	bounds.Xmin = bounds.Xmax = bodies.front().X;           //  Initialize Xmin, Xmax
        for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {   //  Loop over bodies
          bounds.Xmin = min(B->X, bounds.Xmin - 1e-5);          //   Update Xmin
          bounds.Xmax = max(B->X, bounds.Xmax + 1e-5);          //   Update Xmax
        }                                                       //  End loop over bodies
      }                                                         // End if for empty body vector
      logger::stopTimer("Get bounds");                          // Stop timer
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Update Xmin and Xmax of bodies
    Bounds getBounds(Bodies bodies, Bounds bounds) {
      logger::startTimer("Get bounds");                         // Start timer
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {     // Loop over bodies
        bounds.Xmin = min(B->X, bounds.Xmin - 1e-5);            //  Update Xmin
        bounds.Xmax = max(B->X, bounds.Xmax + 1e-5);            //  Update Xmax
      }                                                         // End loop over bodies
      logger::stopTimer("Get bounds");                          // Stop timer
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Get Xmin and Xmax of cells
    Bounds getBounds(Cells cells) {
      logger::startTimer("Get bounds");                         // Start timer
      Bounds bounds;                                            // Bounds : Contains Xmin, Xmax
      if (cells.empty()) {                                      // If cell vector is empty
	bounds.Xmin = bounds.Xmax = 0;                          //  Set bounds to 0
      } else {                                                  // If cell vector is not empty
	bounds.Xmin = bounds.Xmax = cells.front().X;            //  Initialize Xmin, Xmax
        for (C_iter C=cells.begin(); C!=cells.end(); C++) {     //  Loop over cells
          bounds.Xmin = min(C->X - 1e-5, bounds.Xmin);          //   Update Xmin
          bounds.Xmax = max(C->X + 1e-5, bounds.Xmax);          //   Update Xmax
        }                                                       //  End loop over cells
      }                                                         // End if for empty body vector
      logger::stopTimer("Get bounds");                          // Stop timer
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Update Xmin and Xmax of cells
    Bounds getBounds(Cells cells, Bounds bounds) {
      logger::startTimer("Get bounds");                         // Start timer
      for (C_iter C=cells.begin(); C!=cells.end(); C++) {       // Loop over cells
        bounds.Xmin = min(C->X - 1e-5, bounds.Xmin);            //  Update Xmin
        bounds.Xmax = max(C->X + 1e-5, bounds.Xmax);            //  Update Xmax
      }                                                         // End loop over cells
      logger::stopTimer("Get bounds");                          // Stop timer
      return bounds;                                            // Return Xmin and Xmax
    }

    // Global
    // -----------------------------------------
    // TODO: coarse-grained checkin/checkout
    // What are the stuffs like 1e-5...? We cannot parallelize...

    //! Get Xmin and Xmax of bodies
    Bounds getBounds(GBodies bodies) {
      if (ityr::is_master()) {
        logger::startTimer("Get bounds");                         // Start timer
      }
      Bounds bounds;                                            // Bounds : Contains Xmin, Xmax
      if (bodies.empty()) {                                     // If body vector is empty
	bounds.Xmin = bounds.Xmax = 0;                          //  Set bounds to 0
      } else {                                                  // If body vector is not empty
        bounds = ityr::root_exec([=] {
          auto mp_X = static_cast<vec3 Body::*>(&Source::X);
          auto minmax_vec_reducer = ityr::reducer::make_reducer(
              [x = bodies.begin()->*(mp_X)]() {
                return Bounds{x, x};
              },
              [](Bounds& acc, const auto& B) {
                acc.Xmin = min(B.X - 1e-5, acc.Xmin);
                acc.Xmax = max(B.X + 1e-5, acc.Xmax);
              },
              [](Bounds& acc_l, const Bounds& acc_r) {
                acc_l.Xmin = min(acc_l.Xmin, acc_r.Xmin);
                acc_l.Xmax = max(acc_l.Xmax, acc_r.Xmax);
              });
          return ityr::reduce(
              body_par_policy,
              bodies.begin(), bodies.end(),
              minmax_vec_reducer);
        });
      }                                                         // End if for empty body vector
      if (ityr::is_master()) {
        logger::stopTimer("Get bounds");                          // Stop timer
      }
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Update Xmin and Xmax of bodies
    Bounds getBounds(GBodies bodies, Bounds bounds) {
      if (ityr::is_master()) {
        logger::startTimer("Get bounds");                         // Start timer
      }
      Bounds new_bounds = ityr::root_exec([=] {
        auto minmax_vec_reducer = ityr::reducer::make_reducer(
            [=]() { return bounds; },
            [](Bounds& acc, const auto& B) {
              acc.Xmin = min(B.X - 1e-5, acc.Xmin);
              acc.Xmax = max(B.X + 1e-5, acc.Xmax);
            },
            [](Bounds& acc_l, const Bounds& acc_r) {
              acc_l.Xmin = min(acc_l.Xmin, acc_r.Xmin);
              acc_l.Xmax = max(acc_l.Xmax, acc_r.Xmax);
            });
        return ityr::reduce(
            body_par_policy,
            bodies.begin(), bodies.end(),
            minmax_vec_reducer);
      });
      if (ityr::is_master()) {
        logger::stopTimer("Get bounds");                          // Stop timer
      }
      return new_bounds;                                            // Return Xmin and Xmax
    }

    //! Get Xmin and Xmax of cells
    Bounds getBounds(GCells cells) {
      if (ityr::is_master()) {
        logger::startTimer("Get bounds");                         // Start timer
      }
      Bounds bounds;                                            // Bounds : Contains Xmin, Xmax
      if (cells.empty()) {                                      // If cell vector is empty
	bounds.Xmin = bounds.Xmax = 0;                          //  Set bounds to 0
      } else {                                                  // If cell vector is not empty
        bounds = ityr::root_exec([=] {
          auto mp_X = static_cast<vec3 Cell::*>(&CellBase::X);
          auto minmax_vec_reducer = ityr::reducer::make_reducer(
              [x = cells.begin()->*(mp_X)]() {
                return Bounds{x, x};
              },
              [](Bounds& acc, const auto& C) {
                acc.Xmin = min(C.X - 1e-5, acc.Xmin);
                acc.Xmax = max(C.X + 1e-5, acc.Xmax);
              },
              [](Bounds& acc_l, const Bounds& acc_r) {
                acc_l.Xmin = min(acc_l.Xmin, acc_r.Xmin);
                acc_l.Xmax = max(acc_l.Xmax, acc_r.Xmax);
              });
          return ityr::reduce(
              body_par_policy,
              cells.begin(), cells.end(),
              minmax_vec_reducer);
        });
      }                                                         // End if for empty body vector
      if (ityr::is_master()) {
        logger::stopTimer("Get bounds");                          // Stop timer
      }
      return bounds;                                            // Return Xmin and Xmax
    }

    //! Update Xmin and Xmax of cells
    Bounds getBounds(GCells cells, Bounds bounds) {
      if (ityr::is_master()) {
        logger::startTimer("Get bounds");                         // Start timer
      }
      Bounds new_bounds = ityr::root_exec([=] {
        auto minmax_vec_reducer = ityr::reducer::make_reducer(
            [=]() { return bounds; },
            [](Bounds& acc, const auto& C) {
              acc.Xmin = min(C.X - 1e-5, acc.Xmin);
              acc.Xmax = max(C.X + 1e-5, acc.Xmax);
            },
            [](Bounds& acc_l, const Bounds& acc_r) {
              acc_l.Xmin = min(acc_l.Xmin, acc_r.Xmin);
              acc_l.Xmax = max(acc_l.Xmax, acc_r.Xmax);
            });
        return ityr::reduce(
            body_par_policy,
            cells.begin(), cells.end(),
            minmax_vec_reducer);
      });
      if (ityr::is_master()) {
        logger::stopTimer("Get bounds");                          // Stop timer
      }
      return new_bounds;                                            // Return Xmin and Xmax
    }
  };
}
#endif
