/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testExampleDummy.h
 * @brief  example of unit test
 * @author Siyi Hu, Luca Carlone
 */

#include <algorithm>
#include <vector>
#include <string>
#include <cctype>
#include <utility>
#include <iostream>
#include <fstream>
#include <cmath>
#include <gtsam/base/Lie.h>

#include <CppUnitLite/TestHarness.h>

#include "FUSES/StiefelProduct.h"
#include "test_config.h"

using namespace std;
using namespace gtsam;

/* ************************************************************************* */
// Data
static const double tol = 1e-7;

/* ************************************************************************* */
TEST(testUtils, test1){
  StiefelProduct sp = StiefelProduct(4,10);
  Matrix R = sp.random_sample();

  // test project
  Matrix R_expected = R; // already on manifold
  Matrix R_actual = sp.project(R);
  EXPECT(assert_equal(R_expected, R_actual, tol));

  // test actual project
  Matrix R1 = sp.random_sample();
  Matrix R2 = R + R1; // not on manifold

  Matrix R_expected2(10,4);
  for(size_t i=0;i<10;i++){
    double normRow = R2.block(i,0,1,4).norm();
    R_expected2.block(i,0,1,4) = R2.block(i,0,1,4) / normRow;
  }
  Matrix R_actual2 = sp.project(R2);
  EXPECT(assert_equal(R_expected2, R_actual2, tol));
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
