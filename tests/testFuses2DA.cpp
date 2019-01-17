/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testFuses2DA.cpp
 * @brief  unit test for Fuses2DA
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

#include "FUSES/Fuses2DA.h"
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
TEST(testFuses2DA, contructor1){
  cout << "\nTesting constructor1 (This also tests randY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  Fuses2DA fs(crf);

  // check crf
  EXPECT(fs.getCRF().equals(crf, tol));

  // check optiond_.r0
  EXPECT(fs.getOptions().r0 == 2);

  // check dimension of Y_
  auto Y = fs.getY();
  EXPECT(Y.rows() == 7*3+1);
  EXPECT(Y.cols() == 2);

  EXPECT(fs.getOptions().initializationType == FromUnary);

  // convert from -1/+1 to binary matrix
  size_t nrNodes = 7;
  size_t nrClasses = 3;

  EXPECT(Y(nrNodes*nrClasses, 0) == 1); // last row
  EXPECT(Y(nrNodes*nrClasses, 1) == 0);

  Matrix U = Y.block(0,1,nrNodes*nrClasses, 1);
  EXPECT(assert_equal(-Matrix::Zero(nrNodes*nrClasses,1), U, tol));

  Matrix B = (UtilsGM::Reshape(Y.block(0,0,nrNodes*nrClasses, 1), nrClasses,
      nrNodes).transpose() + Matrix::Ones(nrNodes,nrClasses) )/2;

  // check if Y is on the Riemannian manifold
  for(int i=0; i<B.rows(); ++i) {
    EXPECT_DOUBLES_EQUAL(1.0, B.row(i).sum(), tol);
  }

  // check if the unary factors are used in Y
  EXPECT(B(0, 0) == 1);
  EXPECT(B(1, 0) == 1);
  EXPECT(B(2, 1) == 1);
  EXPECT(B(3, 0) == 1);
  EXPECT(B(4, 2) == 1);
}

/* ************************************************************************* */
TEST(testFuses2DA, contructor2){
  cout << "\nTesting constructor1 (This also tests initializeQandPrecon "
      "function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  Fuses2DA fs(crf);
  crf.print();

  // check data matrix Q
  Eigen::SparseMatrix<double> Q = fs.getDataMatrix().selfadjointView<Eigen::Upper>();
  Matrix Qactual = Matrix(Q);
  Matrix Qcompact = (Matrix(10,10)
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

  Matrix Qexpected = Matrix::Zero(crf.nrNodes*crf.nrClasses+1,crf.nrNodes*crf.nrClasses+1);
  // manual computation of kronecker product
  Matrix Haug = Matrix::Zero(crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses);
  for (int i = 0; i < crf.nrNodes; i++){
    for (int j = 0; j < crf.nrNodes; j++){
      Haug.block(i*crf.nrClasses, j*crf.nrClasses,crf.nrClasses,crf.nrClasses)
                                = Qcompact(i,j) * Matrix::Identity(crf.nrClasses,crf.nrClasses);
      Haug.block(j*crf.nrClasses, j*crf.nrClasses,crf.nrClasses,crf.nrClasses)
                                = Qcompact(i,j) * Matrix::Identity(crf.nrClasses,crf.nrClasses);
    }
  }
  Qexpected.block(0,0,crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses) = 0.25*Haug;

  Matrix G = 2.0 * Qcompact.block(0,crf.nrNodes,crf.nrNodes,crf.nrClasses);
  Matrix g = UtilsGM::Reshape(G.transpose(),size_t(crf.nrNodes*crf.nrClasses),size_t(1));

  Matrix g_tilde =  0.5*(Haug*Matrix::Ones(crf.nrNodes*crf.nrClasses,1)) + 0.5*g;

  Qexpected.block(0,crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses,1) =
      0.5 * g_tilde;
  Qexpected.block(crf.nrNodes*crf.nrClasses,0,1,crf.nrNodes*crf.nrClasses) =
      0.5 * g_tilde.transpose();
  EXPECT(assert_equal(Qexpected, Qactual, tol));
}

/* ************************************************************************* */
TEST(testFuses2DA, contructor3){
  cout << "\nTesting constructor2..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 6;
  Fuses2DA fs(crf,options);

  // check crf
  EXPECT(fs.getCRF().equals(crf, tol));

  // check optiond_.r0
  EXPECT(fs.getOptions().r0 == 6);

  // check dimension of Y_
  auto Y = fs.getY();
  EXPECT(Y.rows() == 7*3+1);
  EXPECT(Y.cols() == 6);

  EXPECT(fs.getOptions().initializationType == FromUnary);

  // convert from -1/+1 to binary matrix
  size_t nrNodes = 7;
  size_t nrClasses = 3;
  std::cout << "test last row" <<endl;
  EXPECT(Y(nrNodes*nrClasses, 0) == 1); // last row
  EXPECT(Y(nrNodes*nrClasses, 1) == -1);

  std::cout << "test right column" <<endl;
  Matrix U = Y.block(0,1,nrNodes*nrClasses, 1);
  EXPECT(assert_equal(-Matrix::Ones(nrNodes*nrClasses,1), U, tol));

  std::cout << "test left column" <<endl;
  Matrix B = (UtilsGM::Reshape(Y.block(0,0,nrNodes*nrClasses, 1), nrClasses,
      nrNodes).transpose() + Matrix::Ones(nrNodes,nrClasses) )/2;

  // check if Y is on the Riemannian manifold
  for(int i=0; i<B.rows(); ++i) {
    EXPECT_DOUBLES_EQUAL(1.0, B.row(i).sum(), tol);
  }

  // check if the unary factors are used in Y
  EXPECT(B(0, 0) == 1);
  EXPECT(B(1, 0) == 1);
  EXPECT(B(2, 1) == 1);
  EXPECT(B(3, 0) == 1);
  EXPECT(B(4, 2) == 1);
}

/* ************************************************************************* */
TEST(testFuses2DA, contructor4){
  cout << "\nTesting constructor3..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Y_expected = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5);
  Y_expected.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*UtilsGM::Reshape(Y.block(0,0,crf.nrNodes,crf.nrClasses).transpose(),
          crf.nrNodes*crf.nrClasses,1)
      -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5);
  Y_expected.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  // create Fuses2
  Fuses2DA fs(crf, Y_expected);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 5);

  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses2DA, contructor5){
  cout << "\nTesting constructor4..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Y_expected = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5);
  Y_expected.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*(UtilsGM::Reshape(Y.block(0,0,crf.nrNodes,crf.nrClasses).transpose(),crf.nrNodes*crf.nrClasses,1)-0.5*Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5));
  Y_expected.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  // create Fuses2
  FUSESOpts options;
  options.r0 = 6; // even with incorrect r0, Y dictates right size
  Fuses2DA fs(crf, Y_expected, options);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 5);

  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses2DA, contructor6){
  cout << "\nTesting constructor5 (This also tests setY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Y_expected = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,2); // note: by default the rank is 2!
  Y_expected.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*(UtilsGM::Reshape(Y.block(0,0,crf.nrNodes,crf.nrClasses).transpose(),crf.nrNodes*crf.nrClasses,1)-0.5*Matrix::Ones(crf.nrNodes*crf.nrClasses+1,1));
  Y_expected.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  // create Fuses2
  vector<size_t> Yvect{0, 1, 0, 0, 2, 0, 2};
  Fuses2DA fs(crf, Yvect);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 2);

  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses2DA, solve1){
  cout << "\nTesting solve ..."
       << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options;
  options.verbose = false;
  Fuses2 fs1(crf1,options);
  Matrix initY = fs1.getY();
  fs1.solve();

  Fuses2DA fsDA1(crf1,initY,options); // same initial guess
  fsDA1.max_iterations_DA = 1;
  fsDA1.solveDA();

  EXPECT(assert_equal(fs1.getResult().Yopt , fsDA1.getResult().Yopt));
  DOUBLES_EQUAL(fs1.getResult().SDPval , fsDA1.getResult().SDPval, 1e-5);
}

/* ************************************************************************* */
TEST(testFuses2DA, solve2){
  cout << "\nTesting solve 2 ..."
       << endl;

  // Use data train1500Data
  SegmentationFrontEnd sfe1(train1500Data);
  CRF crf1 = sfe1.getCRF();

  Fuses2 fs1(crf1);
  Matrix initY = fs1.getY();
  fs1.solve();

  Fuses2DA fsDA1(crf1,initY); // same initial guess
  fsDA1.max_iterations_DA = 1;
  fsDA1.solveDA();

  EXPECT(assert_equal(fs1.getResult().Yopt , fsDA1.getResult().Yopt));
  DOUBLES_EQUAL(fs1.getResult().SDPval , fsDA1.getResult().SDPval, 1e-5);
}

/* ************************************************************************* *
TEST(testFuses2DA, solve3){
  cout << "\nTesting solve 3 ..."
       << endl;

  // same as one of previous tests, but also checks dual variable update
  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  Fuses2 fs1(crf1);
  Matrix initY = fs1.Y_;
  fs1.solve();

  Fuses2DA fsDA1(crf1,initY); // same initial guess
  fsDA1.max_iterations_DA = 1;

  // check initial violation
  Matrix z = initY.topRows(crf1.nrNodes*crf1.nrClasses) * initY.bottomRows(1).transpose();
  Matrix Z = UtilsGM::Reshape(z,crf1.nrClasses,crf1.nrNodes).transpose();
  Vector expectedEqViolation = Z.rowwise().sum() - (2-crf1.nrClasses)*Vector::Ones(crf1.nrNodes);
  EXPECT(assert_equal(expectedEqViolation, Vector::Zero(crf1.nrNodes)));

  // now solve using DA!
  fsDA1.solveDA();

  EXPECT(assert_equal(fs1.result_.Yopt , fsDA1.result_.Yopt));
  DOUBLES_EQUAL(fs1.result_.SDPval , fsDA1.result_.SDPval, 1e-5);

  // check initial stepsize
  DOUBLES_EQUAL(0.05 , fsDA1.alpha_init, 1e-7);
  Matrix z_viol =  fsDA1.Y_.topRows(crf1.nrNodes*crf1.nrClasses) * fsDA1.Y_.bottomRows(1).transpose();
  Matrix Z_viol = UtilsGM::Reshape(z_viol,crf1.nrClasses,crf1.nrNodes).transpose();
  Vector expectedEqViolation2 = Z_viol.rowwise().sum() - (2-crf1.nrClasses)*Vector::Ones(crf1.nrNodes);
  EXPECT(assert_equal(expectedEqViolation2, fsDA1.eqViolation));

  // check norm violation
  DOUBLES_EQUAL(expectedEqViolation2.norm() , fsDA1.norm_constraint_violation, 1e-5);

  // check dual variables
  EXPECT(assert_equal(fsDA1.alpha_init * expectedEqViolation2 , fsDA1.y_k));
}

/* ************************************************************************* */
TEST(testFuses2DA, loggin){
  cout << "\nTesting logging ..."
       << endl;

  FUSESOpts options;
  options.verbose = false;

  // same as one of previous tests, but also checks dual variable update
  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  Fuses2DA fsDA1(crf1,options);
  fsDA1.max_iterations_DA = 2;
  fsDA1.solveDA();
}

/* ************************************************************************* */
TEST(testFuses2DA, constraints){
  cout << "\nTesting constraints ..."
       << endl;

  FUSESOpts options;
  options.verbose = false;
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  Fuses2DA fsDA1(crf1,options);
  fsDA1.max_iterations_DA = 2;
  fsDA1.solveDA();

  // check Aeq and beq
  EXPECT(fsDA1.Aeq.rows() == crf1.nrNodes);
  EXPECT(fsDA1.Aeq.cols() == crf1.nrNodes*crf1.nrClasses);
  Matrix X_rounded_m1p1 = (2.0*fsDA1.getResult().xhat -
      Matrix::Ones(crf1.nrNodes,crf1.nrClasses));
  Matrix x_rounded_m1p1 = UtilsGM::Reshape(X_rounded_m1p1.transpose(),
      crf1.nrNodes*crf1.nrClasses,1);

  double norm_violation_rounded = (fsDA1.Aeq * x_rounded_m1p1 - fsDA1.beq).norm();
  DOUBLES_EQUAL(0 , norm_violation_rounded, 1e-5);

  // check beq
  for(size_t i=0; i<fsDA1.beq.size(); i++){
	  DOUBLES_EQUAL(2.0 - crf1.nrClasses , fsDA1.beq(i), 1e-5);
  }

  // check Aeq
  Matrix Aeq_mat = Matrix(fsDA1.Aeq);
  DOUBLES_EQUAL(1.0 , fsDA1.Aeq.coeffRef(0,0), 1e-5);

  Vector u = Vector::Ones(crf1.nrNodes*crf1.nrClasses);
  Vector Aeq_1 =  Aeq_mat * u;
  Vector Aeq_1b =  fsDA1.Aeq * u; //sparse
  EXPECT(assert_equal(Aeq_1 , Aeq_1b));

  for(size_t i=0; i<Aeq_1.size(); i++){
	  DOUBLES_EQUAL(crf1.nrClasses , Aeq_1(i), 1e-5);
  }
  Vector t1_Aeq =  Vector::Ones(crf1.nrNodes).transpose() * fsDA1.Aeq;
  for(size_t i=0; i<t1_Aeq.size(); i++){
	  DOUBLES_EQUAL(1.0 , t1_Aeq(i), 1e-5);
  }
}
/* ************************************************************************* */
TEST(testFuses2DA, solveAgainstSDP0){
  cout << "\nTesting solveAgainstSDP0..." << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(toyCRFfile);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.precon = None;
  options2.initializationType = Random; //IMPORTANT
  options2.log_iterates = true; // IMPORTANT, otherwise no cost correction is possible
  options2.verbose = false;
  Matrix Y0 = (Matrix(22, 2) <<
      -0.950002,  -0.312243,
      0.970599,   0.240704,
      0.999285,  0.0378083,
      -0.435168,   0.900349,
      -0.923819 ,  -0.38283,
      0.0513253 ,  0.998682,
      0.108306  ,-0.994118,
      -0.192709 ,  0.981256,
      0.662928 , -0.748683,
      0.66056 , -0.750773,
      0.391758 , -0.920068,
      -0.89749 , -0.441035,
      -0.997529, -0.0702501,
      0.372856 , -0.927889,
      -0.312863,  -0.949798,
      0.891559 ,  0.452904,
      -0.934846,   0.355054,
      -0.705191,  -0.709018,
      -0.950178,   0.311709,
      -0.123402,  -0.992357,
      0.564344 ,   0.82554,
      0.768824 ,   0.63946).finished();

  Fuses2DA fs1(crf1,options2);

  EXPECT(fs1.getOptions().r0 == 2);
  std::cout << "objective" << fs1.evaluate_objective(Y0) << std::endl;
  std::cout << "fs1.Y_\n" << fs1.getY() << std::endl;

  fs1.max_iterations_DA = 1;
  fs1.solveDA();
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-11.9000, values1.back(), 1e-1); // expected from matlab testFusesDA.m

  // other tests:
  EXPECT_DOUBLES_EQUAL(-4.2250, fs1.offset, tol);

  // check the equaltion we use to get the right-most column of YY'
  auto Y = fs1.getY();
  auto crf = fs1.getCRF();
  Matrix yRightColActual = Y.topRows(crf.nrNodes*crf.nrClasses) * Y.bottomRows(1).transpose();
  Matrix yRightColExpected =
      (Y * Y.transpose()).block(0, crf.nrNodes*crf.nrClasses, crf.nrNodes*crf.nrClasses,1);
  EXPECT(assert_equal(yRightColExpected , yRightColActual));

  // check computation of offset
  // fs1.y_k = Vector::Random(crf1.nrNodes);
  double o_actual = fs1.computeOffset(fs1.y_k, Y);

  Matrix Q_y_k = Matrix::Zero(crf1.nrNodes*crf1.nrClasses+1,crf1.nrNodes*crf1.nrClasses+1);
  Q_y_k.block(0,crf1.nrNodes*crf1.nrClasses,crf1.nrNodes*crf1.nrClasses,1) =
      fs1.Aeq.transpose() * fs1.y_k;
  double o_expected = (Y.transpose() * Q_y_k *Y).trace();
  EXPECT_DOUBLES_EQUAL(o_expected, o_actual, 1e-4);

  // check computation of o_expected in a second way
  Matrix Q_y_k2_tmp = Matrix::Zero(crf1.nrNodes*crf1.nrClasses+1,crf1.nrNodes*crf1.nrClasses+1);
  Q_y_k2_tmp.block(0,crf1.nrNodes*crf1.nrClasses,crf1.nrNodes*crf1.nrClasses,1) =
      fs1.Aeq.transpose() * fs1.y_k;
  Matrix Q_y_k2 = 0.5 * (Q_y_k2_tmp + Q_y_k2_tmp.transpose());
  // Q_y_k2.block(crf1.nrNodes,0,crf1.nrClasses,crf1.nrNodes) = 0.5*(fs1.y_k * Vector::Ones(1,crf1.nrClasses) ).transpose();
  double o_expected2 = (Y.transpose() * Q_y_k2 * Y).trace();
  EXPECT_DOUBLES_EQUAL(o_expected, o_expected2, 1e-4);
}

/* ************************************************************************* */
TEST(testFuses2DA, solveAgainstSDP00){
  cout << "\nTesting solveAgainstSDP00..." << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(toyCRFfile);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.precon = None;
  options2.initializationType = Random; //IMPORTANT
  options2.log_iterates = true; // IMPORTANT, otherwise no cost correction is possible
  options2.verbose = false;

  Fuses2DA fs1(crf1,options2);
  EXPECT(fs1.getOptions().r0 == 2);

  Vector y_k_0 = (Vector(7) <<
         0.3924,
         0.5081,
         0.4428,
         0.4662,
         0.2772,
        -0.0061,
        -0.0061).finished();

  fs1.max_iterations_DA = 1;
  fs1.solveDA(y_k_0);
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-6.3125, values1.back(), 0.1); // expected from matlab testFusesDA.m

  // other tests:
  EXPECT_DOUBLES_EQUAL(-4.2250, fs1.offset, tol);
  for(size_t i = 0; i<5; i++){ // other nodes do not show up in cost
    EXPECT_DOUBLES_EQUAL(y_k_0(i), fs1.y_k(i), 0.1);
  }
}

/* ************************************************************************* */
TEST(testFuses2DA, solveAgainstSDP10){
  cout << "\nTesting solveAgainstSDP10..." << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.precon = None;
  options2.initializationType = Random; //IMPORTANT
  options2.log_iterates = true; // IMPORTANT, otherwise no cost correction is possible
  options2.verbose = true;
  options2.grad_norm_tol = 1e-4;

  Fuses2DA fs1(crf1,options2);
  EXPECT(fs1.getOptions().r0 == 2);

  Vector y_k_0 = (Vector(crf1.nrNodes) <<
          0.1615,
          0.1723,
          0.3948,
          0.3958,
          0.2327,
          0.3951,
          0.1093,
          0.1594,
          0.3937,
          0.2811,
          0.2800,
          0.2639,
          0.2339,
          0.1431,
          0.2821,
          0.3920,
          0.2900,
          0.2851,
          0.2302,
          0.2408,
          0.1571,
          0.3925,
          0.2571,
          0.2864,
          0.2888,
          0.1791,
          0.4410,
          0.2292,
          0.1433,
          0.2779,
          0.4061,
          0.3968,
          0.1446,
          0.2864,
          0.1897,
          0.1870,
          0.1401,
          0.4215,
          0.2696,
          0.3943,
          0.2575,
          0.1973,
          0.2495,
          0.1805,
          0.1915,
          0.3896,
          0.1736,
          0.1712,
          0.1719,
          0.2532,
          0.2092,
          0.2276,
          0.2369,
          0.2174,
          0.3859,
          0.2475,
          0.1982,
          0.2478,
          0.3935,
          0.2504,
          0.4725,
          0.2455,
          0.3928,
          0.2731,
          0.3933,
          0.2497,
          0.2319,
          0.3957,
          0.1877,
          0.2476,
          0.2515,
          0.3959,
          0.2199,
          0.3970,
          0.3063,
          0.3979,
          0.3953,
          0.2530,
          0.2749,
          0.2473,
          0.2999,
          0.3974,
          0.2694,
          0.3041,
          0.2908,
          0.3945,
          0.3946,
          0.3921,
          0.3980,
          0.1365,
          0.3966,
          0.2965).finished();

  fs1.max_iterations_DA = 1;
  fs1.solveDA(y_k_0);
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-65.0884, values1.back(), 0.1); // expected from matlab testFusesDA.m

  // other tests:
  EXPECT_DOUBLES_EQUAL(-63.3882, fs1.offset, 1e-3);
  EXPECT(assert_equal(y_k_0, fs1.y_k, 0.1));
}

/* ************************************************************************* */
TEST(testFuses2DA, solveAgainstSDP){
  cout << "\nTesting solveAgainstSDP1..." << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.precon = None;
  options2.initializationType = Random; //IMPORTANT
  options2.log_iterates = true; // IMPORTANT, otherwise no cost correction is possible
  options2.verbose = true;
  options2.grad_norm_tol = 1e-4;
  Fuses2DA fs1(crf1,options2);
  fs1.alpha_init = 0.05;
  fs1.min_eqViolation_tol = 0.2;

  EXPECT(fs1.getOptions().r0 == 2);

  fs1.solveDA();
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-64.911361349012182, values1.back(), 1e-1); // expected from matlab testFuses2DA.m

  // check computation of offset
  // fs1.y_k = Vector::Random(crf1.nrNodes);
  auto Y = fs1.getY();
  double o_actual = fs1.computeOffset(fs1.y_k, Y);

  Matrix Q_y_k = Matrix::Zero(crf1.nrNodes*crf1.nrClasses+1,crf1.nrNodes*crf1.nrClasses+1);
  Q_y_k.block(0,crf1.nrNodes*crf1.nrClasses,crf1.nrNodes*crf1.nrClasses,1) =
      fs1.Aeq.transpose() * fs1.y_k;
  double o_expected = (Y.transpose() * Q_y_k *Y).trace();
  EXPECT_DOUBLES_EQUAL(o_expected, o_actual, 1e-4);

  // check computation of o_expected in a second way
  Matrix Q_y_k2_tmp = Matrix::Zero(crf1.nrNodes*crf1.nrClasses+1,crf1.nrNodes*crf1.nrClasses+1);
  Q_y_k2_tmp.block(0,crf1.nrNodes*crf1.nrClasses,crf1.nrNodes*crf1.nrClasses,1) =
      fs1.Aeq.transpose() * fs1.y_k;
  Matrix Q_y_k2 = 0.5 * (Q_y_k2_tmp + Q_y_k2_tmp.transpose());
  // Q_y_k2.block(crf1.nrNodes,0,crf1.nrClasses,crf1.nrNodes) = 0.5*(fs1.y_k * Vector::Ones(1,crf1.nrClasses) ).transpose();
  double o_expected2 = (Y.transpose() * Q_y_k2 * Y).trace();
  EXPECT_DOUBLES_EQUAL(o_expected, o_expected2, 1e-4);
}

/* ************************************************************************* */
TEST(testFuses2DA, solveAgainstSDP2){
  cout << "\nTesting solveAgainstSDP2..." << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train200Data);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.precon = None;
  options2.initializationType = Random; //IMPORTANT
  options2.log_iterates = true; // IMPORTANT, otherwise no cost correction is possible
  options2.grad_norm_tol = 1e-4;
  //options2.verbose = true;
  Fuses2DA fs1(crf1,options2);
  fs1.alpha_init = 0.05;
  fs1.min_eqViolation_tol = 0.2;

  fs1.solveDA();
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-1.511532517546024e+02, values1.back(), 0.5); // expected from matlab testFusesDA.m
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
