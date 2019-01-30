/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   Timer.h
 * @brief  functions to measure elapsed computation time
 * @author Siyi Hu, Luca Carlone
 */

#ifndef Timer_H_
#define Timer_H_

#include <chrono>

namespace myUtils{
  namespace Timer {

  inline std::chrono::high_resolution_clock::time_point tic() {
    return std::chrono::high_resolution_clock::now();
  }

  /** When this function is called with a chrono::time_point struct returned by
   * tick(), it returns the elapsed time (in seconds) between the calls to
   * tick() and tock().*/
  inline double toc(const std::chrono::high_resolution_clock::time_point&
      time_tic) {
    std::chrono::high_resolution_clock::time_point time_now =
        std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(time_now -
        time_tic).count();
  }

  } // namespace Timer
} // namespace myUtils

#endif /* Timer_H_ */


