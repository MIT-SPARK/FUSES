/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testFuses.cpp
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

#include "FUSES/Fuses2.h"
#include "FUSES/Fuses.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/StiefelMixedProduct.h"
#include "FUSES/UtilsOpenCV.h"
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
TEST(testFuses2, contructor1){
  cout << "\nTesting constructor1 (This also tests randY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf);

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

  Matrix U = Y.block(0, 1, nrNodes*nrClasses, 1);
  EXPECT(assert_equal(-Matrix::Zero(nrNodes*nrClasses,1), U, tol));

  Matrix B = (UtilsGM::Reshape(Y.block(0,0,nrNodes*nrClasses, 1), nrClasses,
      nrNodes).transpose()
  + Matrix::Ones(nrNodes,nrClasses) )/2;

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
TEST(testFuses, contructor2){
  cout << "\nTesting constructor1 (This also tests initializeQandPrecon "
      "function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf);
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
TEST(testFuses, contructor3){
  cout << "\nTesting constructor2..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 6;
  Fuses2 fs(crf,options);

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
TEST(testFuses, contructor4){
  cout << "\nTesting constructor3..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Y_expected = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5);
  Y_expected.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*(UtilsGM::Reshape(Y.block(0,0,crf.nrNodes,crf.nrClasses).transpose(),crf.nrNodes*crf.nrClasses,1)-0.5*Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5));
  Y_expected.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  // create Fuses2
  Fuses2 fs(crf, Y_expected);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 5);

  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses, contructor5){
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
  Fuses2 fs(crf, Y_expected, options);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 5);

  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses, contructor6){
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
  Fuses2 fs(crf, Yvect);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 2);

  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses, contructor7){
  cout << "\nTesting constructor6 (This also tests setY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Y_expected = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5); // note: we change the rank to 5 in fuses below
  Y_expected.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*(UtilsGM::Reshape(Y.block(0,0,crf.nrNodes,crf.nrClasses).transpose(),crf.nrNodes*crf.nrClasses,1)-0.5*Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5));
  Y_expected.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  // create Fuses2
  vector<size_t> Yvect{0, 1, 0, 0, 2, 0, 2};
  FUSESOpts options;
  options.r0 = 5;
  Fuses2 fs(crf, Yvect, options);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 5);

  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses, relaxation_rank){
  cout << "\nTesting set_relaxation_rank and get_relaxation_rank..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Y_expected = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5);
  Y_expected.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*(UtilsGM::Reshape(Y.block(0,0,crf.nrNodes,crf.nrClasses).transpose(),crf.nrNodes*crf.nrClasses,1)-0.5*Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5));
  Y_expected.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  // create Fuses2
  Fuses2 fs1(crf, Y_expected);
  EXPECT(fs1.get_relaxation_rank() == 5); // should be the same as Y.cols()

  fs1.set_relaxation_rank(4);
  EXPECT(fs1.get_relaxation_rank() == 4);

  Fuses2 fs2(crf);
  EXPECT(fs2.get_relaxation_rank() == 2);
}

/* ************************************************************************* */
TEST(testFuses, evaluate_objective_and_round){
  cout << "\nTesting evaluate_objective0..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Y_expected = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,1);
  Y_expected.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*UtilsGM::Reshape(Y.block(0,0,crf.nrNodes,crf.nrClasses).transpose(),
          crf.nrNodes*crf.nrClasses,1)
      -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5);
  Y_expected.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  // create Fuses2
  Fuses2 fs(crf, Y_expected);

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

  // test matrices again
  Matrix G = 2.0 * Qcompact.block(0,crf.nrNodes,crf.nrNodes,crf.nrClasses);
  Matrix g = UtilsGM::Reshape(G.transpose(),size_t(crf.nrNodes*crf.nrClasses),size_t(1));

  Matrix g_tilde =  0.5*(Haug*Matrix::Ones(crf.nrNodes*crf.nrClasses,1)) + 0.5*g;

  Qexpected.block(0,crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses,1) =
      0.5 * g_tilde;
  Qexpected.block(crf.nrNodes*crf.nrClasses,0,1,crf.nrNodes*crf.nrClasses) =
      0.5 * g_tilde.transpose();
  EXPECT(assert_equal(Qexpected, Qactual, tol));

  // get binary matrix and test rounding
  Matrix B = fs.round_solution(Y_expected);
  EXPECT(assert_equal(Y.block(0,0,7,3), B, tol));

  Matrix BI = Matrix::Zero(crf.nrNodes+crf.nrClasses,crf.nrClasses);
  BI.block(0,0,crf.nrNodes,crf.nrClasses) = B;
  BI.block(crf.nrNodes,0,crf.nrClasses,crf.nrClasses) = Matrix::Identity(crf.nrClasses,crf.nrClasses);
  EXPECT(assert_equal(Y, BI, tol));

  // create Fuses
  Fuses fs1(crf,Y);
  Matrix Ylabels = Y.block(0,0,7,3); // take the top part of Y, containing the labels
  EXPECT_DOUBLES_EQUAL(crf.evaluate(Ylabels), fs1.evaluate_objective(Y), tol);

  // check matrix fuses
  Eigen::SparseMatrix<double> Qfs1 = fs1.getDataMatrix().selfadjointView<Eigen::Upper>();
  Matrix Qfs1Actual = Matrix(Qfs1);
  EXPECT(assert_equal(Qcompact, Qfs1Actual, tol));

  EXPECT_DOUBLES_EQUAL(crf.evaluate(Ylabels), (Y.transpose() * Qcompact * Y).trace(), tol);
  EXPECT_DOUBLES_EQUAL(crf.evaluate(Ylabels), (BI.transpose() * Qcompact * BI).trace(), tol);

  // now check fuses2
  EXPECT(fs.getY().cols() == 1);
  Eigen::SparseMatrix<double> Qfs2 = fs.getDataMatrix().selfadjointView<Eigen::Upper>();
  Matrix Qfs2Actual = Matrix(Qfs2);
  Matrix Z =  fs.getY() * fs.getY().transpose();
  // verify:
  // CHECK COST WITHOUT OFFSET (MANUALLY)
  // H_tilde = 1/4*Haug;
  // g_tilde = 1/2*(Haug*ones(N*K,1)) + 1/2*g;
  // offset = 1/4*sum(Haug(:)) + 1/2*sum(g);
  // trace(H_tilde*Z(1:N*K, 1:N*K)) + g_tilde'*Z(1:N*K, N*K+1)
  EXPECT_DOUBLES_EQUAL( (0.25 * Haug * Z.block(0,0,crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses)).trace()
      + ( g_tilde.transpose() *  Z.block(0,crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses,1) ).trace() ,
      (fs.getY().transpose() * (Qfs2Actual * fs.getY())).trace(), tol);

  // CHECK COST
  Vector u = Vector::Ones(crf.nrNodes*crf.nrClasses);
  double myOffset = 0.25*(u.transpose() * Haug * u).trace() + 0.5*(u.transpose() * g).trace();
  //std::cout << "first term " << (fs.Y_.transpose() * (Qfs2Actual * fs.Y_)).trace() << std::endl;
  //std::cout << "myOffset " << myOffset << std::endl;
  EXPECT_DOUBLES_EQUAL( myOffset , fs.offset, tol);
  //  EXPECT_DOUBLES_EQUAL(crf.evaluate(Ylabels), fs.evaluate_objective(fs.Y_), tol);

  // CHECK COST WITH MANUAL OFFSET COMPUTATION
  fs.offset = 0.0;
  EXPECT_DOUBLES_EQUAL( (0.25 * Haug * Z.block(0,0,crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses)).trace()
      + ( g_tilde.transpose() *  Z.block(0,crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses,1) ).trace() ,
      fs.evaluate_objective(fs.getY()), tol);
  // NOTE: sum in eigen cannot be used: the following 2 fail
  //  EXPECT_DOUBLES_EQUAL(
  //      ( Vector::Ones(1,crf.nrNodes*crf.nrClasses) * Haug * Vector::Ones(crf.nrNodes*crf.nrClasses,1)).trace() ,
  //    Haug.sum(), tol);
  //  EXPECT_DOUBLES_EQUAL(
  //      ( Vector::Ones(1,crf.nrNodes*crf.nrClasses) * g).trace() ,
  //      g.sum(), tol);

  Eigen::SparseMatrix<double> Q2 = fs.getDataMatrix().selfadjointView<Eigen::Upper>();
  Matrix Qactual2 = Matrix(Q2);

  EXPECT_DOUBLES_EQUAL( crf.evaluate(Ylabels) ,
      fs.evaluate_objective(fs.getY()) + myOffset, tol);
}

/* ************************************************************************* */
TEST(testFuses, evaluate_objective){
  cout << "\nTesting evaluate_objective..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  // crf.print();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                               1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Y_expected = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,1);
  Y_expected.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*(UtilsGM::Reshape(Y.block(0,0,crf.nrNodes,crf.nrClasses).transpose(),crf.nrNodes*crf.nrClasses,1)-0.5*Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5));
  Y_expected.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  Fuses2 fs(crf, Y_expected);

  // F(Y) = Q*Y*Y'
  EXPECT_DOUBLES_EQUAL(-3.65, fs.evaluate_objective(Y_expected), tol);
  EXPECT_DOUBLES_EQUAL(-3.65, fs.evaluate_objective(fs.getY()), tol);

  Matrix Ylabels = Y.block(0,0,7,3); // take the top part of Y, containing the labels
  EXPECT_DOUBLES_EQUAL(crf.evaluate(Ylabels), fs.evaluate_objective(Y_expected), tol);
  EXPECT_DOUBLES_EQUAL(crf.evaluate(Ylabels), fs.evaluate_objective(fs.getY()), tol);
}

/* ************************************************************************* */
TEST(testFuses, Euclidean_gradient){
  cout << "\nTesting Euclidean_gradient..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses2 fs(crf);
  Matrix Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,1,-1,-1,-1,
                               1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
                               -1,-1,-1,-1,1,-1, 1,-1,-1,-1,-1,-1,
                               -1,-1,-1,-1,1,-1, 1,-1).finished();
  // Rexpected = 2*Q*Y
  Matrix Rexpected = (Matrix(22, 2) << -0.65,    0.9,
                                       -0.25,    0.4,
                                           0,    0.4,
                                      -1.175,  1.175,
                                           0,  0.675,
                                           0,  0.675,
                                      -0.175,    0.5,
                                      -0.725,      1,
                                        -0.1,    0.5,
                                      -0.825,  1.075,
                                        -0.2,  0.575,
                                       -0.05,  0.575,
                                       -0.15,   0.15,
                                           0,   0.15,
                                        -0.5,   0.65,
                                           0,      0,
                                           0,      0,
                                           0,      0,
                                           0,      0,
                                           0,      0,
                                           0,      0,
                                        0.65,   5.95).finished();
  EXPECT(assert_equal(Rexpected, fs.Euclidean_gradient(Y), tol));
}

/* ************************************************************************* */
TEST(testFuses, Riemannian_gradient1){
  cout << "\nTesting Riemannian_gradient with one input..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses2 fs(crf);

  // compute using Y and nablaF_Y
  Matrix Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,1,-1,-1,-1,
                                 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
                                 -1,-1,-1,-1,1,-1, 1,-1,-1,-1,-1,-1,
                                 -1,-1,-1,-1,1,-1, 1,-1).finished();
  Matrix Ractual = fs.Riemannian_gradient(Y);
  Matrix Rexpected = (Matrix(22, 2) << 0.9,                     -0.65,
                                      -0.4,                      0.25,
                                      -0.4,                         0,
                                    -1.175,                     1.175,
                                     0.675,                         0,
                                    -0.675,                         0,
                                       0.5,                    -0.175,
                                        -1,                     0.725,
                                      -0.5,                       0.1,
                                     1.075,                    -0.825,
                                    -0.575,                       0.2,
                                    -0.575,                      0.05,
                                     -0.15,                      0.15,
                                     -0.15,                         0,
                                      0.65,                      -0.5,
                                         0,                         0,
                                         0,                         0,
                                         0,                         0,
                                         0,                         0,
                                         0,                         0,
                                         0,                         0,
                                      5.95,                      0.65).finished();
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, Riemannian_gradient2){
  cout << "\nTesting Riemannian_gradient with two inputs..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses2 fs(crf);

  // compute using Y and nablaF_Y
  Matrix Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,1,-1,-1,-1,
                               1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
                               -1,-1,-1,-1,1,-1, 1,-1,-1,-1,-1,-1,
                               -1,-1,-1,-1,1,-1, 1,-1).finished();
  Matrix nablaF_Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,2,-1,3,-1,
                                      1,-1,2,-1,0,-1, 1,-1,9,-1,-1,-1,
                                      2,-1,-1,-1,1,-1, 1,-1,0,-1,1,-1,
                                      3,-1,1,-1,0,-1, 1,-1).finished();
  Matrix Ractual = fs.Riemannian_gradient(Y, nablaF_Y);
  Matrix Rexpected = (Matrix(22, 2) << -1,     1,
                                        1,     1,
                                        1,     1,
                                        1,     1,
                                       -1,     2,
                                        1,    -3,
                                       -1,     1,
                                        1,    -2,
                                        1,     0,
                                       -1,     1,
                                        1,    -9,
                                        1,     1,
                                        1,    -2,
                                        1,     1,
                                       -1,     1,
                                       -1,     1,
                                        1,     0,
                                        1,    -1,
                                        1,    -3,
                                        1,    -1,
                                       -1,     0,
                                       -1,     1).finished();
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, Euclidean_Hessian_vector_product){
  cout << "\nTesting Euclidean_Hessian_vector_product..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses2 fs(crf);
  Matrix Ydot = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,1,-1,-1,-1,
                               1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
                               -1,-1,-1,-1,1,-1, 1,-1,-1,-1,-1,-1,
                               -1,-1,-1,-1,1,-1, 1,-1).finished();
  // Rexpected = 2*Q*Ydot
  Matrix Rexpected = (Matrix(22, 2) << -0.65,    0.9,
                                       -0.25,    0.4,
                                           0,    0.4,
                                      -1.175,  1.175,
                                           0,  0.675,
                                           0,  0.675,
                                      -0.175,    0.5,
                                      -0.725,      1,
                                        -0.1,    0.5,
                                      -0.825,  1.075,
                                        -0.2,  0.575,
                                       -0.05,  0.575,
                                       -0.15,   0.15,
                                           0,   0.15,
                                        -0.5,   0.65,
                                           0,      0,
                                           0,      0,
                                           0,      0,
                                           0,      0,
                                           0,      0,
                                           0,      0,
                                        0.65,   5.95).finished();
  EXPECT(assert_equal(Rexpected, fs.Euclidean_gradient(Ydot), tol));
}

/* ************************************************************************* */
TEST(testFuses, Riemannian_Hessian_vector_product1){
  cout << "\nTesting Riemannian_Hessian_vector_product with one input..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses2 fs(crf);
  Matrix Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,1,-1,-1,-1,
                                1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
                                -1,-1,-1,-1,1,-1, 1,-1,-1,-1,-1,-1,
                                -1,-1,-1,-1,1,-1, 1,-1).finished();
 Matrix nablaF_Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,2,-1,3,-1,
                                     1,-1,2,-1,0,-1, 1,-1,9,-1,-1,-1,
                                     2,-1,-1,-1,1,-1, 1,-1,0,-1,1,-1,
                                     3,-1,1,-1,0,-1, 1,-1).finished();
  Matrix Ydot = (Matrix(22, 2) << 1,0,0,0,0,0, -1,0,0,0,0,0,
                                  0,0,-1,0,2,0, 1,0,0,0,1,0,
                                  0,0,0,0,0,0, 1,0,0,0,0,0,
                                  3,0,-1,0,0,0, 0,0).finished();

  // compare matrix values
  Matrix Rexpected = (Matrix(22, 2) <<  0,                     -1.95,
                                        0,                         0,
                                        0,                     0.075,
                                        0,                    -1.775,
                                        0,                    0.1125,
                                        0,                     0.325,
                                        0,                     0.025,
                                        0,                         1,
                                        0,                    2.0875,
                                        0,                    -1.975,
                                        0,                   -0.0875,
                                        0,                     2.175,
                                        0,                     0.025,
                                        0,                     -0.05,
                                        0,                    -0.125,
                                        0,                        -2,
                                        0,                         0,
                                        0,                         0,
                                        0,                        -6,
                                        0,                         0,
                                        0,                         0,
                                        0,                   -0.6875).finished();
  Matrix Ractual = fs.Riemannian_Hessian_vector_product(Y, nablaF_Y, Ydot);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, Riemannian_Hessian_vector_product2){
  cout << "\nTesting Riemannian_Hessian_vector_product with two inputs..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses2 fs(crf);
  Matrix Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,1,-1,-1,-1,
                               1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
                              -1,-1,-1,-1,1,-1, 1,-1,-1,-1,-1,-1,
                              -1,-1,-1,-1,1,-1, 1,-1).finished();
  Matrix nablaF_Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,2,-1,3,-1,
                                      1,-1,2,-1,0,-1, 1,-1,9,-1,-1,-1,
                                      2,-1,-1,-1,1,-1, 1,-1,0,-1,1,-1,
                                      3,-1,1,-1,0,-1, 1,-1).finished();
  Matrix Ydot = (Matrix(22, 2) << 1,0,0,0,0,0, -1,0,0,0,0,0,
                                  0,0,-1,0,2,0, 1,0,0,0,1,0,
                                  0,0,0,0,0,0, 1,0,0,0,0,0,
                                  3,0,-1,0,0,0, 0,0).finished();

  // compare matrix values
  Matrix Rexpected = (Matrix(22, 2) <<  0,                       1.6,
                                        0,                         0,
                                        0,                     0.075,
                                        0,                     0.225,
                                        0,                    0.1125,
                                        0,                     0.325,
                                        0,                     0.025,
                                        0,                     0.275,
                                        0,                   -0.7125,
                                        0,                     1.925,
                                        0,                   -0.0875,
                                        0,                     -0.35,
                                        0,                     0.025,
                                        0,                     -0.05,
                                        0,                    -0.125,
                                        0,                         0,
                                        0,                         0,
                                        0,                         0,
                                        0,                         0,
                                        0,                         0,
                                        0,                         0,
                                        0,                   -0.6875).finished();
  Matrix Ractual = fs.Riemannian_Hessian_vector_product(Y, Ydot);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, tangent_space_projection){
  cout << "\nTesting tangent_space_projection..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses2 fs(crf);
  Matrix Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,1,-1,-1,-1,
                                 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
                                -1,-1,-1,-1,1,-1, 1,-1,-1,-1,-1,-1,
                                -1,-1,-1,-1,1,-1, 1,-1).finished();
  Matrix Ydot = (Matrix(22, 2) << 1,0,0,0,0,0, -1,0,0,0,0,0,
                                  0,0,-1,0,2,0, 1,0,0,0,1,0,
                                  0,0,0,0,0,0, 1,0,0,0,0,0,
                                  3,0,-1,0,0,0, 0,0).finished();

  // compare matrix values
  // compare matrix values
    Matrix Rexpected = (Matrix(22, 2) <<  0,     1,
                                          0,     0,
                                          0,     0,
                                          0,     1,
                                          0,     0,
                                          0,     0,
                                          0,     0,
                                          0,     1,
                                          0,    -2,
                                          0,     1,
                                          0,     0,
                                          0,    -1,
                                          0,     0,
                                          0,     0,
                                          0,     0,
                                          0,     1,
                                          0,     0,
                                          0,     0,
                                          0,    -3,
                                          0,     1,
                                          0,     0,
                                          0,     0).finished();
  Matrix Ractual = fs.tangent_space_projection(Y, Ydot);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, retract){
  cout << "\nTesting retract..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses2 fs(crf);
  Matrix Y = (Matrix(22, 2) << 1,-1,-1,-1,-1,-1, -1,-1,1,-1,-1,-1,
                                 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
                                -1,-1,-1,-1,1,-1, 1,-1,-1,-1,-1,-1,
                                -1,-1,-1,-1,1,-1, 1,-1).finished();
  Matrix Ydot = (Matrix(22, 2) << 1,0,0,0,0,0, -1,0,0,0,0,0,
                                  0,0,-1,0,2,0, 1,0,0,0,1,0,
                                  0,0,0,0,0,0, 1,0,0,0,0,0,
                                  3,0,-1,0,0,0, 0,0).finished();

  // compare matrix values
  Matrix Rexpected = (Matrix(22, 2) <<  2/sqrt(5),    -1/sqrt(5),
                                       -1/sqrt(2),    -1/sqrt(2),
                                       -1/sqrt(2),    -1/sqrt(2),
                                       -2/sqrt(5),    -1/sqrt(5),
                                        1/sqrt(2),    -1/sqrt(2),
                                       -1/sqrt(2),    -1/sqrt(2),
                                        1/sqrt(2),    -1/sqrt(2),
                                       -2/sqrt(5),    -1/sqrt(5),
                                        1/sqrt(2),    -1/sqrt(2),
                                        2/sqrt(5),    -1/sqrt(5),
                                       -1/sqrt(2),    -1/sqrt(2),
                                                0,            -1,
                                       -1/sqrt(2),    -1/sqrt(2),
                                       -1/sqrt(2),    -1/sqrt(2),
                                        1/sqrt(2),    -1/sqrt(2),
                                        2/sqrt(5),    -1/sqrt(5),
                                       -1/sqrt(2),    -1/sqrt(2),
                                       -1/sqrt(2),    -1/sqrt(2),
                                        2/sqrt(5),    -1/sqrt(5),
                                       -2/sqrt(5),    -1/sqrt(5),
                                        1/sqrt(2),    -1/sqrt(2),
                                        1/sqrt(2),    -1/sqrt(2)).finished();
  Matrix Ractual = fs.retract(Y, Ydot);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, round_solution){
  cout << "\nTesting round_solution..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF(); // copying the crf instead of creating a const ref

  FUSESOpts options;
  options.r0 = 1;
  Fuses2 fs(crf, options);
  Matrix Y = (Matrix(10, 4) << 0,   0.9,   0.1,   0.2,
                            -0.1,   0.2,   0.8,     0,
                             0.3,     0,     1,  -0.1,
                               0,  0.01,   0.8,  -0.9,
                               1,     0,     0,     0,
                               0,     2,     0,   0.5,
                               0,     0,  0.99, -0.01,
                               1,     0,     0,     0,
                             1.5,     0,  -0.5,     0,
                               0,   0.5,     0,     1).finished();
  // Toy example is not compatible with the matrix Y above
  crf.nrNodes = 7;
  crf.nrClasses = 3;

  Matrix Z_ur = Y.topRows(crf.nrNodes) * Y.bottomRows(crf.nrClasses).transpose();
  //std::cout << "Z_ur \n " << Z_ur << std::endl;

  Matrix Y_m1p1 = -Matrix::Ones(crf.nrNodes*crf.nrClasses+1,1);
  Y_m1p1.block(0,0,crf.nrNodes*crf.nrClasses,1) =
      2*(UtilsGM::Reshape(Z_ur.transpose(),crf.nrNodes*crf.nrClasses,1)-0.5*Matrix::Ones(crf.nrNodes*crf.nrClasses+1,5));
  Y_m1p1.bottomLeftCorner(1,1) = Matrix::Identity(1,1);

  // std::cout << "Y_m1p1 \n " << Y_m1p1 << std::endl;

  // compare matrix values
  Matrix Rexpected = (Matrix(7, 3) << 0,     0,     1,
                                      0,     0,     1,
                                      1,     0,     0,
                                      1,     0,     0,
                                      0,     1,     0,
                                      0,     0,     1,
                                      1,     0,     0).finished();
  // NOTE: rounded estimate is a binary matrix!
  Matrix Ractual_m1p1 = fs.round_solution(Y_m1p1);
  EXPECT(assert_equal(Rexpected, Ractual_m1p1, tol));
}

/* ************************************************************************* */
TEST(testFuses, round_solution2){
  cout << "\nTesting round_solution..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 1;
  Fuses2 fs1(crf, options);
  Matrix Zexpected = fs1.getY() * fs1.getY().transpose();
  Matrix zexpected = Zexpected.block(0,crf.nrNodes*crf.nrClasses,crf.nrNodes*crf.nrClasses,1);
  Matrix zactual = fs1.getY().topRows(crf.nrNodes*crf.nrClasses) * fs1.getY().bottomRows(1).transpose();
  EXPECT(assert_equal(zexpected, zactual, tol));
}

/* ************************************************************************* *
TEST(testFuses, compute_Q_minus_Lambda){
  cout << "\nTesting compute_Q_minus_Lambda..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  Fuses fs(crf);
  Matrix Y = (Matrix(10, 4) << 0,   0.9,   0.1,   0.2,
                            -0.1,   0.2,   0.8,     0,
                             0.3,     0,     1,  -0.1,
                               0,   0.1,   0.8,  -0.9,
                               1,     0,     0,     0,
                               0,     2,     0,   0.5,
                               0,     0,   0.9,  -0.1,
                               1,     0,     0,     0,
                             1.5,     0,  -0.5,     0,
                               0,   0.5,     0,     1).finished();

  // compare matrix values
  Matrix Rexpected = (Matrix(10, 10)
      <<  0.0635,   -0.25,       0,   -0.15,       0,
                        0,       0,    -0.5,       0,       0,
           -0.25, 0.32025,  -0.225,    -0.2,       0,
                        0,       0,    -0.5,       0,       0,
               0,  -0.225,   0.334,  -0.175,    -0.1,
                        0,       0,       0,    -0.5,       0,
           -0.15,    -0.2,  -0.175, 0.28625,   -0.05,
                        0,       0,    -0.5,       0,       0,
               0,       0,    -0.1,   -0.05,    0.03,
                        0,       0,       0,       0,    -0.5,
               0,       0,       0,       0,       0,
                        0,       0,       0,       0,       0,
               0,       0,       0,       0,       0,
                        0,       0,       0,       0,       0,
            -0.5,    -0.5,       0,    -0.5,       0,
                        0,       0,   -0.05,  -0.175,   0.225,
               0,       0,    -0.5,       0,       0,
                        0,       0,  -0.175,  -0.025,    0.35,
               0,       0,       0,       0,    -0.5,
                        0,       0,   0.225,    0.35,       0).finished();
  Matrix Ractual = fs.compute_Q_minus_Lambda(Y);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* *
TEST(testFuses, compute_Q_minus_Lambda_min_eig){
  cout << "\nTesting compute_Q_minus_Lambda_min_eig..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses fs(crf);
  Matrix Y = (Matrix(10, 4) << 0,   0.9,   0.1,   0.2,
                            -0.1,   0.2,   0.8,     0,
                             0.3,     0,     1,  -0.1,
                               0,   0.1,   0.8,  -0.9,
                               1,     0,     0,     0,
                               0,     2,     0,   0.5,
                               0,     0,   0.9,  -0.1,
                               1,     0,     0,     0,
                             1.5,     0,  -0.5,     0,
                               0,   0.5,     0,     1).finished();

  // expected min eigenvalue and eigenvector
  double min_eigenvalue_expected = -1.0982200542902;
  Vector min_eigenvector_expected = (Vector(10) << -0.397213253143113,
                                                    -0.37226889270197,
                                                   -0.191264212603003,
                                                   -0.346622729787022,
                                                    0.080163671071766,
                                                                    0,
                                                                    0,
                                                   -0.632779938725315,
                                                   -0.275058658989099,
                                                    0.253799638156682).finished();

  // results
  double min_eigenvalue_actual;
  Vector min_eigenvector_actual;
  EXPECT(fs.compute_Q_minus_Lambda_min_eig(Y, min_eigenvalue_actual,
      min_eigenvector_actual));

  // compare eigenvalue and eigenvector
  EXPECT_DOUBLES_EQUAL(min_eigenvalue_expected, min_eigenvalue_actual, tol);
  EXPECT(assert_equal(min_eigenvector_expected, min_eigenvector_actual, tol));
}

/* ************************************************************************* */
TEST(testFuses, solveToy){
  cout << "\nTesting solve..." << endl;
  SegmentationFrontEnd sfe1(toyCRFfile);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.r0 = 2;
  options2.precon = None;
  options2.initializationType = RandomClasses;
  options2.verbose = true;
  Fuses2 fs2(crf1,options2);
  Matrix Y = (Matrix(22, 2) << -0.7241,   -0.6897,
     -0.1881,   -0.9821,
     -0.7445,   -0.6676,
     -0.5745,    0.8185,
     -0.3242,    0.9460,
     -0.8791,    0.4766,
     -0.9915,    0.1302,
     -0.4553,   -0.8903,
     -0.9402,    0.3405,
     -0.8313,   -0.5558,
     -0.8150,    0.5795,
     -0.5937,   -0.8047,
     -0.1758,   -0.9844,
     -0.6854,    0.7282,
     -0.5956,    0.8033,
     -0.8931,    0.4498,
     -0.9265,    0.3763,
     -0.9617,   -0.2743,
     -0.5010,    0.8654,
     -0.2955,   -0.9553,
     -0.3313,    0.9435,
     -0.3764,   -0.9264).finished();
  fs2.setY(Y);

  // check initial cost:
  EXPECT_DOUBLES_EQUAL(-2.07264196075,
		  fs2.evaluate_objective(fs2.getY()) - fs2.offset, tol); // expected from matlab toyTest.m

  fs2.solve();

  // check objective computation:
  Matrix xsol = Matrix::Ones(22,1);
  Matrix X01 = fs2.getResult().xhat;
  Matrix X_m1p1 = 2.0*X01 - Matrix::Ones(crf1.nrNodes,crf1.nrClasses);
  xsol.block(0,0,21,1) = UtilsGM::Reshape(X_m1p1.transpose(),21,1);
  EXPECT_DOUBLES_EQUAL(fs2.evaluate_objective(xsol),
		  fs2.getResult().Fxhat, tol); // expected from matlab toyTest.m

  // check solution against matlab sdp:
  Matrix Zexpected = fs2.getY() * fs2.getY().transpose();
  std::cout << "Zexpected \n " << Zexpected << std::endl;
  std::cout << "Y_ \n " << fs2.getY() << std::endl;
  std::cout << "xhat01 " << UtilsGM::Reshape(X01.transpose(),21,1) << std::endl;

  // check results against matlab
  const vector<double>& values2 = fs2.getResult().function_values.back();
  UtilsOpenCV::PrintVector<double>(values2,"values2");
  //std::cout << "values2 " << values2.back() + fs2.offset << std::endl;
  //std::cout << "Fxhat2 " << fs2.result_.Fxhat << std::endl;
  EXPECT_DOUBLES_EQUAL(-11.9000, values2.back(), 1e-4); // expected from matlab toyTest.m
  // expected does not match actual: however this crf is underconstrained,
  // so there might be multiple optimal solutions, which is why the following fails
  // NOTE: optimality of the SDP does not mean equal cost for the rounded
  // EXPECT_DOUBLES_EQUAL(-6.2000, fs2.result_.Fxhat, tol); // expected from matlab toyTest.m
  // std::cout << "offset" << fs2.offset << std::endl;

  // check Fuses
  FUSESOpts options1;
  options1.precon = None;
  Fuses fs1(crf1,options1);
  fs1.solve();
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-6.2000, fs1.getResult().Fxhat, tol); // expected from matlab toyTest.m
  EXPECT_DOUBLES_EQUAL(-6.6117, values1.back(), 1e-4); // expected from matlab toyTest.m
}

/* ************************************************************************* */
TEST(testFuses, solve){
  cout << "\nTesting solve..." << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  FUSESOpts options2;
  options2.r0 = 2;
  options2.precon = None;
  options2.initializationType = RandomClasses;
  options2.verbose = true;
  Fuses2 fs1(crf1,options2);
  fs1.solve();
  const vector<double>& values1 = fs1.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-225.552950, values1.back(), 1e-2); // expected from matlab testFromCSV.m
}

/* ************************************************************************* */
TEST(testFuses, solve1){
  cout << "\nTesting solve (This only test the solution not the status)..."
       << endl;

  FUSESOpts options;
  options.r0 = 2;
  options.precon = None;
  options.initializationType = Random; // IMPORTANT!!

  // Use data with 100 nodes
  {
  SegmentationFrontEnd sfe(train100Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-225.552950, values.back(), 1e-2); // expected from matlab testFromCSV.m
  }

  // Use data with 200 nodes
  {
  SegmentationFrontEnd sfe(train200Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-534.005379, values.back(), 1e-2); // expected from matlab testFromCSV.m
  }

  // Use data with 500 nodes
  {
  SegmentationFrontEnd sfe(train500Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-1356.728428, values.back(), 1e-2); // expected from matlab testFromCSV.m
  }

  // Use data with 800 nodes
  {
  SegmentationFrontEnd sfe(train800Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-2144.095737, values.back(), 1e-2); // expected from matlab testFromCSV.m
  }

  // Use data with 1500 nodes
  {
  SegmentationFrontEnd sfe(train1500Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-4052.008815, values.back(), 1e-1); // expected from matlab testFromCSV.m
  }
}

/* ************************************************************************* */
TEST(testFuses, solve2){
  cout << "\nTesting solve with Jacobi precon "
       << "(This only test the solution not the status)..."
       << endl;

  FUSESOpts options;
  options.r0 = 2;
  options.precon = Jacobi; //  !!!!!
  options.initializationType = Random; // IMPORTANT!!

  // Use data with 100 nodes
  {
  SegmentationFrontEnd sfe(train100Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-225.552950, values.back(), 1e-2); // expected from matlab testFromCSV.m
  }

  // Use data with 200 nodes
  {
  SegmentationFrontEnd sfe(train200Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-534.005379, values.back(), 1e-2); // expected from matlab testFromCSV.m
  }

  // Use data with 500 nodes
  {
  SegmentationFrontEnd sfe(train500Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-1356.728428, values.back(), 1e-2); // expected from matlab testFromCSV.m
  }

  // Use data with 800 nodes
  {
  SegmentationFrontEnd sfe(train800Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-2144.095737, values.back(), 1e-2); // expected from matlab testFromCSV.m
  }

  // Use data with 1500 nodes
  {
  SegmentationFrontEnd sfe(train1500Data);
  CRF crf = sfe.getCRF();
  Fuses2 fs(crf,options);
  fs.solve();
  const vector<double>& values = fs.getResult().function_values.back();
  EXPECT_DOUBLES_EQUAL(-4052.008815, values.back(), 1); // expected from matlab testFromCSV.m
  }
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
