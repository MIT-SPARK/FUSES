/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testStiefelMixedProduct.cpp
 * @brief  unit test for StiefelMixedProduct
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

#include "FUSES/StiefelMixedProduct.h"
#include "test_config.h"

using namespace gtsam;
using namespace std;

/* ************************************************************************* */
// Data
static const double tol = 1e-7;

/* ************************************************************************* */
TEST(testFuses, contructor){
  cout << "\nTesting constructor..." << endl;

  // default constructor
  StiefelMixedProduct sp1;
  sp1.set_n(3);
  EXPECT_DOUBLES_EQUAL(sp1.get_n(), 3, tol);

  // constructor with r, n, k inputs
  StiefelMixedProduct sp2(5, 10, 4);
  EXPECT_DOUBLES_EQUAL(sp2.get_r(), 5, tol);
  EXPECT_DOUBLES_EQUAL(sp2.get_n(), 10, tol);
  EXPECT_DOUBLES_EQUAL(sp2.get_k(), 4, tol);
}

/* ************************************************************************* */
TEST(testFuses, project){
  cout << "\nTesting project..." << endl;

  // small test case
  StiefelMixedProduct sp1(2, 1, 1);
  Matrix A = (Matrix(2, 2) << 1, 2, 0, 1).finished();
  Matrix Qactual = sp1.project(A);
  Matrix Qexpected = (Matrix(2, 2) << 1/sqrt(5), 2/sqrt(5), 0, 1).finished();
  EXPECT(assert_equal(Qexpected, Qactual, tol));


  // bigger test case
  StiefelMixedProduct sp2(4, 5, 3);
  Matrix B = (Matrix(8, 4) << 2,  3,  5,  1,
                             -2,  1, -1,  2,
                              0, -1,  2,  0,
                             -1, -2, -3,  0,
                              1,  0,  0,  0,
                              2,  1,  0,  1,
                              3, -1, -1,  1,
                              0,  2,  0,  0).finished();
  Matrix Ractual = sp2.project(B);
  Matrix Rexpected = (Matrix(8, 4) << 2/sqrt(39),  3/sqrt(39),  5/sqrt(39),  1/sqrt(39),
                                     -2/sqrt(10),  1/sqrt(10), -1/sqrt(10),  2/sqrt(10),
                                      0,          -1/sqrt(5),   2/sqrt(5),   0,
                                     -1/sqrt(14), -2/sqrt(14), -3/sqrt(14),  0,
                                      1,           0,           0,           0,
                                      0.537478608101935,  0.365171517917838,
                                        0.537478608101935,  0.537478608101935,
                                      0.737491285823938, -0.294389157833909,
                                       -0.604149500675936,  0.066670892574001,
                                      0.023594120027977,  0.883167473501726,
                                       -0.423619475471981, -0.200012677722002).finished();
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, SymBlockDiagProduct){
  cout << "\nTesting SymBlockDiagProduct..." << endl;

  // small test case
  StiefelMixedProduct sp1(2, 1, 2);
  Matrix A1 = (Matrix(3, 2) <<  1,  2,  0,  1, -2,  1).finished();
  Matrix B1 = (Matrix(3, 2) <<  3, -1,  1,  5, -1,  6).finished();
  Matrix C1 = (Matrix(3, 2) << -1, -3,  6,  1, -2,  3).finished();

  Matrix Qactual = sp1.SymBlockDiagProduct(A1, B1, C1);
  Matrix Qexpected = (Matrix(3, 2) << -1, -3, 21, 18.5, 11, 28.5).finished();
  EXPECT(assert_equal(Qexpected, Qactual, tol));


  // bigger test case
  StiefelMixedProduct sp2(4, 4, 3);
  Matrix A2 = (Matrix(7, 4) << 2,  3,  5,  1,
                              -2,  1, -1,  2,
                               0, -1,  2,  0,
                              -1, -2, -3,  0,
                               2,  1,  0,  1,
                               3, -1, -1,  1,
                               0,  2,  0,  0).finished();
  Matrix B2 = (Matrix(7, 4) << 2, -1,  6,  2,
                              -2,  1, -1,  2,
                               3, -1,  2,  1,
                               1, -2,  6,  1,
                              -1,  7,  2,  2,
                               1, -5, -1,  3,
                               0,  2,  3, -1).finished();
  Matrix C2 = (Matrix(7, 4) << 2,  2,  0,  2,
                               0, -1, -2,  5,
                               2, -1,  3,  1,
                               1, -2,  6,  1,
                               1,  3,  5,  2,
                               1, -2,  0,  3,
                               3,  2,  4, -4).finished();

  Matrix Ractual = sp2.SymBlockDiagProduct(A2, B2, C2);
  Matrix Rexpected = (Matrix(7, 4) << 66,  66,   0,  66,
                                       0, -10, -20,  50,
                                      10,  -5,  15,   5,
                                     -15,  30, -90, -15,
                                    24.5,  46,  65, -31,
                                     -17, -55, -57,  58,
                                    11.5, 46.5, 53.5, -25).finished();
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, Proj){
  cout << "\nTesting Proj..." << endl;

  // small test case
  StiefelMixedProduct sp1(2, 1, 2);
  Matrix Y1 = (Matrix(3, 2) <<  1,  0,  1,  0,  0,  1).finished();
  Matrix V1 = (Matrix(3, 2) <<  2, -1,  1,  0, -1,  3).finished();

  Matrix Qactual = sp1.Proj(Y1, V1);
  Matrix Qexpected = (Matrix(3, 2) << 0, -1, 0, 0.5, -0.5, 0).finished();
  EXPECT(assert_equal(Qexpected, Qactual, tol));


  // bigger test case
  StiefelMixedProduct sp2(4, 4, 3);
  Matrix Y2 = (Matrix(7, 4) << 1,  0,  0,  0,
                               0,  1,  0,  0,
                               0,  1,  0,  0,
                               1,  0,  0,  0,
                               0,  1,  0,  0,
                               0,  0,  1,  0,
                               0,  0,  0,  1).finished();
  Matrix V2 = (Matrix(7, 4) << 2, -1,  6,  2,
                              -2,  1, -1,  2,
                               3, -1,  2,  1,
                               1, -2,  6,  1,
                              -1,  7,  2,  2,
                               1, -5, -1,  3,
                               0,  2,  3, -1).finished();

  Matrix Ractual = sp2.Proj(Y2, V2);
  Matrix Rexpected = (Matrix(7, 4) << 0,  -1,   6,   2,
                                     -2,   0,  -1,   2,
                                      3,   0,   2,   1,
                                      0,  -2,   6,   1,
                                     -1,   0, 3.5,   0,
                                      1,-3.5,   0,   0,
                                      0,   0,   0,   0).finished();
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, retract){
  cout << "\nTesting retract..." << endl;

  // small test case
  StiefelMixedProduct sp1(2, 1, 1);
  Matrix Y1 = (Matrix(2, 2) << 1, 0, 0, 1).finished();
  Matrix V1 = (Matrix(2, 2) << 0, 2, 0, 0).finished();

  Matrix Qactual = sp1.retract(Y1, V1);
  Matrix Qexpected = (Matrix(2, 2) << 1/sqrt(5), 2/sqrt(5), 0, 1).finished();
  EXPECT(assert_equal(Qexpected, Qactual, tol));


  // bigger test case
  StiefelMixedProduct sp2(4, 5, 3);
  Matrix Y2 = (Matrix(8, 4) << 0,  1,  0,  0,
                               0,  1,  0,  0,
                               1,  0,  0,  0,
                               0,  0,  1,  0,
                               1,  0,  0,  0,
                               0,  1,  0,  0,
                               0,  0,  1,  0,
                               0,  0,  0,  1).finished();
  Matrix V2 = (Matrix(8, 4) << 2,  2,  5,  1,
                              -2,  0, -1,  2,
                              -1, -1,  2,  0,
                              -1, -2, -4,  0,
                               0,  0,  0,  0,
                               2,  0,  0,  1,
                               3, -1, -2,  1,
                               0,  2,  0, -1).finished();
  Matrix Ractual = sp2.retract(Y2, V2);
  Matrix Rexpected = (Matrix(8, 4) << 2/sqrt(39),  3/sqrt(39),  5/sqrt(39),  1/sqrt(39),
                                     -2/sqrt(10),  1/sqrt(10), -1/sqrt(10),  2/sqrt(10),
                                      0,          -1/sqrt(5),   2/sqrt(5),   0,
                                     -1/sqrt(14), -2/sqrt(14), -3/sqrt(14),  0,
                                      1,           0,           0,           0,
                                      0.537478608101935,  0.365171517917838,
                                        0.537478608101935,  0.537478608101935,
                                      0.737491285823938, -0.294389157833909,
                                       -0.604149500675936,  0.066670892574001,
                                      0.023594120027977,  0.883167473501726,
                                       -0.423619475471981, -0.200012677722002).finished();
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, random_sample){
  cout << "\nTesting random_sample..." << endl;

  // small test case
  StiefelMixedProduct sp1(2, 1, 1);
  Matrix M1 = sp1.random_sample();
  Matrix R1 = M1 * M1.transpose();

  EXPECT_DOUBLES_EQUAL(R1(0, 0), 1, tol);
  EXPECT_DOUBLES_EQUAL(R1(1, 1), 1, tol);

  // bigger test case
  StiefelMixedProduct sp2(4, 5, 3);
  Matrix M2 = sp2.random_sample();
  Matrix R2 = M2 * M2.transpose();
  EXPECT_DOUBLES_EQUAL(R2(0, 0), 1, tol);
  EXPECT_DOUBLES_EQUAL(R2(1, 1), 1, tol);
  EXPECT_DOUBLES_EQUAL(R2(2, 2), 1, tol);
  EXPECT_DOUBLES_EQUAL(R2(3, 3), 1, tol);
  EXPECT_DOUBLES_EQUAL(R2(4, 4), 1, tol);
  EXPECT_DOUBLES_EQUAL(R2(4, 4), 1, tol);

  Matrix Qactual = R2.block(5, 5, 3, 3);
  Matrix Qexpected = Matrix::Identity(3, 3);
  EXPECT(assert_equal(Qexpected, Qactual, tol));
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
