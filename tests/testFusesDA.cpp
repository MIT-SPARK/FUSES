/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testFusesDA.cpp
 * @brief  unit test for Fuses
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

#include "FUSES/FusesDA.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/StiefelMixedProduct.h"
#include "test_config.h"

using namespace std;

static const string toyCRFfile = string(DATASET_PATH) + "/test.csv";
static const string train100Data = string(DATASET_PATH) + "/train100Data.csv";
static const string train100Result = string(DATASET_PATH) + "/train100Result.csv";
static const string train200Data = string(DATASET_PATH) + "/train200Data.csv";
static const string train200Result = string(DATASET_PATH) + "/train200Result.csv";
static const string train500Data = string(DATASET_PATH) + "/train500Data.csv";
static const string train500Result = string(DATASET_PATH) + "/train500Result.csv";
static const string train800Data = string(DATASET_PATH) + "/train800Data.csv";
static const string train800Result = string(DATASET_PATH) + "/train800Result.csv";
static const string train1500Data = string(DATASET_PATH) + "/train1500Data.csv";
static const string train1500Result = string(DATASET_PATH) + "/train1500Result.csv";
static const string trainImage = string(DATASET_PATH) + "/trainImage.jpg";
static const string trainLabel = string(DATASET_PATH) + "/trainLabel.png";
static const string param_defaultFile = string(DATASET_PATH) + "/fusesParameters_default.yaml";

using namespace gtsam;
using namespace CRFsegmentation;
using namespace FUSES;

/* ************************************************************************* */
// Data
static const double tol = 1e-7;

/* ************************************************************************* */
TEST(testFusesDA, contructor1){
  cout << "\nTesting constructor1 (This also tests randY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  FusesDA fs(crf);

  // check crf
  EXPECT(fs.getCRF().equals(crf, tol));

  // check optiond_.r0
//  EXPECT(fs.options_.r0 == 4);

  // check dimension of Y_
  auto Y = fs.getY();
  EXPECT(Y.rows() == 10);
//  EXPECT(fs.Y_.cols() == 4);

  // check if Y is on the Riemannian manifold
  for(int i=0; i<Y.rows(); ++i) {
    EXPECT(Y.row(i).norm() == 1);
  }

  // check if the unary factors are used in Y
  EXPECT(Y(0, 0) == 1);
  EXPECT(Y(1, 0) == 1);
  EXPECT(Y(2, 1) == 1);
  EXPECT(Y(3, 0) == 1);
  EXPECT(Y(4, 2) == 1);
}

/* ************************************************************************* */
TEST(testFusesDA, contructor2){
  cout << "\nTesting constructor1 (This also tests initializeQandPrecon function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  FusesDA fs(crf);
  // crf.print();

  // check data matrix Q
  Eigen::SparseMatrix<double> Q = fs.getDataMatrix().selfadjointView<Eigen::Upper>();
  Matrix Qactual = Matrix(Q);
  Matrix Qexpected = (Matrix(10,10)
    << 0,  -0.25,      0,  -0.15,      0,      0,      0,   -0.5,      0,      0,
   -0.25,      0, -0.225,   -0.2,      0,      0,      0,   -0.5,      0,      0,
       0, -0.225,      0, -0.175,   -0.1,      0,      0,      0,   -0.5,      0,
   -0.15,   -0.2, -0.175,      0,  -0.05,      0,      0,   -0.5,      0,      0,
       0,      0,   -0.1,  -0.05,      0,      0,      0,      0,      0,   -0.5,
       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
       0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
    -0.5,   -0.5,      0,   -0.5,      0,      0,      0,      0,      0,      0,
       0,      0,   -0.5,      0,      0,      0,      0,      0,      0,      0,
       0,      0,      0,      0,   -0.5,      0,      0,      0,      0,      0).finished();
  EXPECT(assert_equal(Qexpected, Qactual, tol));
}

/* ************************************************************************* */
TEST(testFusesDA, contructor3){
  cout << "\nTesting constructor2..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 6;
  FusesDA fs(crf, options);

  // check options_.r0
    EXPECT(fs.getOptions().r0 == 6);

    // check dimension of Y_
    auto Y = fs.getY();
    EXPECT(Y.rows() == 10);
    EXPECT(Y.cols() == 6);

    // check if Y is on the Riemannian manifold
    for(int i=0; i<Y.rows(); ++i) {
      EXPECT(Y.row(i).norm() == 1);
    }

    // check if the unary factors are used in Y
    EXPECT(Y(0, 0) == 1);
    EXPECT(Y(1, 0) == 1);
    EXPECT(Y(2, 1) == 1);
    EXPECT(Y(3, 0) == 1);
    EXPECT(Y(4, 2) == 1);
}

/* ************************************************************************* */
TEST(testFusesDA, contructor4){
  cout << "\nTesting constructor3..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                               1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  FusesDA fs(crf, Y);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 3);

  // check Y_
  Matrix Y_expected = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFusesDA, contructor5){
  cout << "\nTesting constructor4..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 6;
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                               1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  FusesDA fs(crf, Y, options);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 3);

  // check Y_
  Matrix Y_expected = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFusesDA, contructor6){
  cout << "\nTesting constructor5 (This also tests setY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  vector<size_t> Y{0, 0, 1, 2, 1, 0, 0};
  FusesDA fs(crf, Y);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 4);

  // check Y_
  Matrix Y_expected = (Matrix(10, 4) << 1,0,0,0, 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,1,0,0,
                                      1,0,0,0, 1,0,0,0, 1,0,0,0, 0,1,0,0, 0,0,1,0).finished();
  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFusesDA, solve1){
  cout << "\nTesting solve ..."
       << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  Fuses fs1(crf1);
  Matrix initY = fs1.getY();
  fs1.solve();

  FusesDA fsDA1(crf1,initY); // same initial guess
  fsDA1.max_iterations_DA = 1;
  fsDA1.solveDA();

  EXPECT(assert_equal(fs1.getResult().Yopt , fsDA1.getResult().Yopt));
  DOUBLES_EQUAL(fs1.getResult().SDPval , fsDA1.getResult().SDPval, 1e-5);
}

/* ************************************************************************* */
TEST(testFusesDA, solve2){
  cout << "\nTesting solve 2 ..."
       << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train1500Data);
  CRF crf1 = sfe1.getCRF();

  Fuses fs1(crf1);
  Matrix initY = fs1.getY();
  fs1.solve();

  FusesDA fsDA1(crf1,initY); // same initial guess
  fsDA1.max_iterations_DA = 1;
  fsDA1.solveDA();

  EXPECT(assert_equal(fs1.getResult().Yopt , fsDA1.getResult().Yopt));
  DOUBLES_EQUAL(fs1.getResult().SDPval , fsDA1.getResult().SDPval, 1e-5);
}


/* ************************************************************************* */
TEST(testFusesDA, solve3){
  cout << "\nTesting solve 3 ..."
       << endl;

  // same as one of previous tests, but also checks dual variable update
  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  Fuses fs1(crf1);
  Matrix initY = fs1.getY();
  fs1.solve();

  FusesDA fsDA1(crf1,initY); // same initial guess
  fsDA1.max_iterations_DA = 1;
  fsDA1.solveDA();

  EXPECT(assert_equal(fs1.getResult().Yopt , fsDA1.getResult().Yopt));
  DOUBLES_EQUAL(fs1.getResult().SDPval , fsDA1.getResult().SDPval, 1e-5);

  // check initial stepsize
  DOUBLES_EQUAL(0.05 , fsDA1.alpha_init, 1e-7);

  // check violation
  Matrix Z = fsDA1.getResult().Yopt * fsDA1.getResult().Yopt.transpose();
  Matrix X = Z.block(0,crf1.nrNodes,crf1.nrNodes,crf1.nrClasses);

  Vector expectedEqViolation = X.rowwise().sum() - Vector::Ones(crf1.nrNodes);
  EXPECT(assert_equal(expectedEqViolation, fsDA1.eqViolation));

  // check norm violation
  DOUBLES_EQUAL(expectedEqViolation.norm() , fsDA1.norm_constraint_violation, 1e-5);

  // check dual variables
  EXPECT(assert_equal(fsDA1.alpha_init * expectedEqViolation , fsDA1.y_k));
}

/* ************************************************************************* */
TEST(testFusesDA, loggin){
  cout << "\nTesting logging ..."
       << endl;

  FUSESOpts options;
  options.verbose = false;

  // same as one of previous tests, but also checks dual variable update
  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  FusesDA fsDA1(crf1,options);
  fsDA1.max_iterations_DA = 2;
  fsDA1.solveDA();
}

/* ************************************************************************* */
TEST(testFusesDA, constraints){
  cout << "\nTesting constraints ..."
       << endl;

  FUSESOpts options;
  options.verbose = false;
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  FusesDA fsDA1(crf1,options);
  fsDA1.max_iterations_DA = 2;
  fsDA1.solveDA();

  // check that a binary solution satisfies the constraints
  Matrix X_rounded = fsDA1.getResult().xhat;

  gtsam::Vector ones_N = gtsam::Vector::Ones(crf1.nrNodes);
  gtsam::Vector ones_K = gtsam::Vector::Ones(crf1.nrClasses);
  double norm_violation_rounded = ((X_rounded * ones_K) - ones_N).norm();
  DOUBLES_EQUAL(0 , norm_violation_rounded, 1e-5);
}

/* ************************************************************************* */
TEST(testFusesDA, solveAgainstSDP){
  cout << "\nTesting solveAgainstSDP..." << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.precon = None;
  options2.initializationType = RandomClasses;
  options2.log_iterates = true; // IMPORTANT, otherwise no cost correction is possible
  //options2.verbose = true;
  options2.grad_norm_tol = 1e-4;
  FusesDA fs1(crf1,options2);

//  EXPECT(fs1.options_.r0 == crf1.nrClasses+1); // nrClasses + 1

  fs1.solveDA();
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-65.707793293609924, values1.back(), 1e-1); // expected from matlab testFusesDA.m

  // check computation of offset
  // fs1.y_k = Vector::Random(crf1.nrNodes);
  double o_actual = fs1.computeOffset(fs1.y_k, fs1.getY());

  Matrix Q_y_k = Matrix::Zero(crf1.nrNodes+crf1.nrClasses,crf1.nrNodes+crf1.nrClasses);
  Q_y_k.block(0,crf1.nrNodes,crf1.nrNodes,crf1.nrClasses) = fs1.y_k *  Matrix::Ones(1,crf1.nrClasses);
  double o_expected = (fs1.getY().transpose() * Q_y_k *fs1.getY()).trace();
  EXPECT_DOUBLES_EQUAL(o_expected, o_actual, 1e-4);

  // check computation of o_expected in a second way
  Matrix Q_y_k2_tmp = Matrix::Zero(crf1.nrNodes+crf1.nrClasses,crf1.nrNodes+crf1.nrClasses);
  Q_y_k2_tmp.block(0,crf1.nrNodes,crf1.nrNodes,crf1.nrClasses) = fs1.y_k * Matrix::Ones(1,crf1.nrClasses);
  Matrix Q_y_k2 = 0.5 * (Q_y_k2_tmp + Q_y_k2_tmp.transpose());
  // Q_y_k2.block(crf1.nrNodes,0,crf1.nrClasses,crf1.nrNodes) = 0.5*(fs1.y_k * Vector::Ones(1,crf1.nrClasses) ).transpose();
  double o_expected2 = (fs1.getY().transpose() * Q_y_k2 * fs1.getY()).trace();
  EXPECT_DOUBLES_EQUAL(o_expected, o_expected2, 1e-4);
}

/* ************************************************************************* */
TEST(testFusesDA, solveAgainstSDP2){
  cout << "\nTesting solveAgainstSDP2..." << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train200Data);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.precon = None;
  options2.initializationType = RandomClasses;
  options2.log_iterates = true; // IMPORTANT, otherwise no cost correction is possible
  //options2.verbose = true;
  FusesDA fs1(crf1,options2);

  fs1.solveDA();
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-1.532055272892281e+02, values1.back(), 0.5); // expected from matlab testFusesDA.m
}

/* ************************************************************************* */
int main() {
  srand(0); // fix randomization seed
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
