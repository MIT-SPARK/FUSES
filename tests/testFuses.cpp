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

#include "FUSES/Fuses.h"
#include "FUSES/SegmentationFrontEnd.h"
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
TEST(testFuses, contructor1){
  cout << "\nTesting constructor1 (This also tests randY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  Fuses fs(crf);

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
TEST(testFuses, contructor2){
  cout << "\nTesting constructor1 (This also tests initializeQandPrecon function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  Fuses fs(crf);

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
TEST(testFuses, contructor3){
  cout << "\nTesting constructor2..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 6;
  Fuses fs(crf, options);

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
TEST(testFuses, contructor4){
  cout << "\nTesting constructor3..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                               1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Fuses fs(crf, Y);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 3);

  // check Y_
  Matrix Y_expected = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses, contructor5){
  cout << "\nTesting constructor4..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 6;
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                               1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Fuses fs(crf, Y, options);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 3);

  // check Y_
  Matrix Y_expected = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                      1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses, contructor6){
  cout << "\nTesting constructor5 (This also tests setY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  vector<size_t> Y{0, 0, 1, 2, 1, 0, 0};
  Fuses fs(crf, Y);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 4);

  // check Y_
  Matrix Y_expected = (Matrix(10, 4) << 1,0,0,0, 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,1,0,0,
                                      1,0,0,0, 1,0,0,0, 1,0,0,0, 0,1,0,0, 0,0,1,0).finished();
  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

/* ************************************************************************* */
TEST(testFuses, contructor7){
  cout << "\nTesting constructor6 (This also tests setY function)..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 5;
  vector<size_t> Y{0, 0, 1, 2, 1, 0, 0};
  Fuses fs(crf, Y, options);

  // check options_.r0
  EXPECT(fs.getOptions().r0 == 5);

  // check Y_
  Matrix Y_expected = (Matrix(10, 5) << 1,0,0,0,0, 1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,1,0,0,0,
                                      1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0).finished();
  EXPECT(assert_equal(Y_expected, fs.getY(), tol));
}

// TODO: the two functions tested here are currently not used
///* ************************************************************************* */
//TEST(testFuses, modifyCRF){
//  cout << "\nTesting setReducedClassConversion and recoverLabels..." << endl;
//  CRF crf;
//  crf.nrClasses = 6;
//  UnaryFactor u1, u2, u3, u4, u5;
//  u1.node = 0;
//  u1.label = 0;
//  u1.weight = 0.1;
//  crf.unaryFactors.push_back(u1);
//
//  u2.node = 0;
//  u2.label = 2;
//  u2.weight = 0.8;
//  crf.unaryFactors.push_back(u2);
//
//  u3.node = 1;
//  u3.label = 5;
//  u3.weight = 1.0;
//  crf.unaryFactors.push_back(u3);
//
//  u4.node = 2;
//  u4.label = 4;
//  u4.weight = 0.9;
//  crf.unaryFactors.push_back(u4);
//
//  u5.node = 3;
//  u5.label = 5;
//  u5.weight = 0.5;
//  crf.unaryFactors.push_back(u5);
//
//  Fuses fs(crf);
//
//  // check modified crf
//  EXPECT(fs.crf_.nrClasses == 4);
//
//  EXPECT(fs.crf_.unaryFactors[0].node == 0);
//  EXPECT(fs.crf_.unaryFactors[0].label == 0);
//  EXPECT(fs.crf_.unaryFactors[0].weight == 0.1);
//
//  EXPECT(fs.crf_.unaryFactors[1].node == 0);
//  EXPECT(fs.crf_.unaryFactors[1].label == 1);
//  EXPECT(fs.crf_.unaryFactors[1].weight == 0.8);
//
//  EXPECT(fs.crf_.unaryFactors[2].node == 1);
//  EXPECT(fs.crf_.unaryFactors[2].label == 3);
//  EXPECT(fs.crf_.unaryFactors[2].weight == 1.0);
//
//  EXPECT(fs.crf_.unaryFactors[3].node == 2);
//  EXPECT(fs.crf_.unaryFactors[3].label == 2);
//  EXPECT(fs.crf_.unaryFactors[3].weight == 0.9);
//
//  EXPECT(fs.crf_.unaryFactors[4].node == 3);
//  EXPECT(fs.crf_.unaryFactors[4].label == 3);
//  EXPECT(fs.crf_.unaryFactors[4].weight == 0.5);
//
//  std::vector<size_t> labels{0, 1, 3, 2, 3, 1, 1, 0, 0, 3};
//  std::vector<size_t> labels_recovered{0, 2, 5, 4, 5, 2, 2, 0, 0, 5};
//  fs.recoverLabels(labels);
//
//  for(int i=0; i<10; ++i) {
//    EXPECT(labels[i] == labels_recovered[i]);
//  }
//}

/* ************************************************************************* */
TEST(testFuses, relaxation_rank){
  cout << "\nTesting set_relaxation_rank and get_relaxation_rank..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                             1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Fuses fs1(crf, Y);
  EXPECT(fs1.get_relaxation_rank() == 3); // should be the same as Y.cols()
  fs1.set_relaxation_rank(4);
  EXPECT(fs1.get_relaxation_rank() == 4);

  Fuses fs2(crf);
//  EXPECT(fs2.get_relaxation_rank() == 4);
}

/* ************************************************************************* */
TEST(testFuses, evaluate_objective){
  cout << "\nTesting evaluate_objective..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  // Y is the label matrix, concatenated with the identity as required by FUSES
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                             1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();

  Fuses fs(crf, Y);

  // F(Y) = Q*Y*Y'
  EXPECT_DOUBLES_EQUAL(-3.65, fs.evaluate_objective(Y), tol);
  EXPECT_DOUBLES_EQUAL(-3.65, fs.evaluate_objective(fs.getY()), tol);

  Matrix Ylabels = Y.block(0,0,7,3); // take the top part of Y, containing the labels
  EXPECT_DOUBLES_EQUAL(crf.evaluate(Ylabels), fs.evaluate_objective(Y), tol);
}

/* ************************************************************************* */
TEST(testFuses, Euclidean_gradient){
  cout << "\nTesting Euclidean_gradient..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  Fuses fs(crf);
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                             1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();

  // Rexpected = 2*Q*Y
  Matrix Rexpected = (Matrix(10, 3) << -1.3,  -0.5,     0,
                                      -2.35,     0,     0,
                                      -0.35, -1.45,  -0.2,
                                      -1.65,  -0.4,  -0.1,
                                       -0.3,     0,    -1,
                                          0,     0,     0,
                                          0,     0,     0,
                                         -2,    -1,     0,
                                         -1,     0,     0,
                                          0,     0,    -1).finished();
  EXPECT(assert_equal(Rexpected, fs.Euclidean_gradient(Y), tol));
}

/* ************************************************************************* */
TEST(testFuses, Riemannian_gradient1){
  cout << "\nTesting Riemannian_gradient with one input..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 3;
  Fuses fs(crf, options);

  // compute using Y and nablaF_Y
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                               1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Ractual = fs.Riemannian_gradient(Y);
  Matrix Rexpected = (Matrix(10, 3) << 0,  -0.5,     0,
                                   -2.35,     0,     0,
                                       0, -1.45,  -0.2,
                                       0,  -0.4,  -0.1,
                                    -0.3,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0).finished();
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, Riemannian_gradient2){
  cout << "\nTesting Riemannian_gradient with two inputs..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  Fuses fs(crf);

  // set dimensions of the Stiefel manifolds
  fs.shp_.set_r(4);
  fs.shp_.set_n(4);
  fs.shp_.set_k(3);

  // compute using Y and nablaF_Y
  Matrix Y = (Matrix(7, 4) <<  1,  0,  0,  0,
                               0,  1,  0,  0,
                               0,  1,  0,  0,
                               1,  0,  0,  0,
                               0,  1,  0,  0,
                               0,  0,  1,  0,
                               0,  0,  0,  1).finished();
  Matrix nablaF_Y = (Matrix(7, 4) << 2, -1,  6,  2,
                                    -2,  1, -1,  2,
                                     3, -1,  2,  1,
                                     1, -2,  6,  1,
                                    -1,  7,  2,  2,
                                     1, -5, -1,  3,
                                     0,  2,  3, -1).finished();
  Matrix Ractual = fs.Riemannian_gradient(Y, nablaF_Y);
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
TEST(testFuses, Euclidean_Hessian_vector_product){
  cout << "\nTesting Euclidean_Hessian_vector_product..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  Fuses fs(crf);
  Matrix Ydot = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();

  // Rexpected = 2*Q*Ydot
  Matrix Rexpected = (Matrix(10, 3) << -1.3,  -0.5,     0,
                                      -2.35,     0,     0,
                                      -0.35, -1.45,  -0.2,
                                      -1.65,  -0.4,  -0.1,
                                       -0.3,     0,    -1,
                                          0,     0,     0,
                                          0,     0,     0,
                                         -2,    -1,     0,
                                         -1,     0,     0,
                                          0,     0,    -1).finished();
  EXPECT(assert_equal(Rexpected, fs.Euclidean_Hessian_vector_product(Ydot), tol));
}

/* ************************************************************************* */
TEST(testFuses, Riemannian_Hessian_vector_product1){
  cout << "\nTesting Riemannian_Hessian_vector_product with one input..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 3;               // set r=3 for the Stiefel product manifold
  Fuses fs(crf, options);
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                             1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix nablaF_Y = (Matrix(10, 3) << -1.3,  -0.5,     0,
                                     -2.35,     0,     0,
                                     -0.35, -1.45,  -0.2,
                                     -1.65,  -0.4,  -0.1,
                                      -0.3,     0,    -1,
                                         0,     0,     0,
                                         0,     0,     0,
                                        -2,    -1,     0,
                                        -1,     0,     0,
                                         0,     0,    -1).finished();
  Matrix Ydot = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();

  // compare matrix values
  Matrix Rexpected = (Matrix(10, 3) << 0,  -0.5,     0,
                                   -2.35,     0,     0,
                                       0, -1.45,  -0.2,
                                       0,  -0.4,  -0.1,
                                    -0.3,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0).finished();
  Matrix Ractual = fs.Riemannian_Hessian_vector_product(Y, nablaF_Y, Ydot);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, Riemannian_Hessian_vector_product2){
  cout << "\nTesting Riemannian_Hessian_vector_product with two inputs..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 3;               // set r=3 for the Stiefel product manifold
  Fuses fs(crf, options);
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                             1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Ydot = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();

  // compare matrix values
  Matrix Rexpected = (Matrix(10, 3) << 0,  -0.5,     0,
                                   -2.35,     0,     0,
                                       0, -1.45,  -0.2,
                                       0,  -0.4,  -0.1,
                                    -0.3,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0).finished();
  Matrix Ractual = fs.Riemannian_Hessian_vector_product(Y, Ydot);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, tangent_space_projection){
  cout << "\nTesting tangent_space_projection..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 3;               // set r=3 for the Stiefel product manifold
  Fuses fs(crf, options);
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                             1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Ydot = (Matrix(10, 3) << 1,0,0, 1,0,0, 2,0,0, 0,0,0, 0,0,1,
                                0,0,0, 0,0,1, 3,0,0, 0,1,0, 0,1,1).finished();

  // compare matrix values
  Matrix Rexpected = (Matrix(10, 3) << 0,     0,     0,
                                       1,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,     0,
                                       0,     0,  -0.5,
                                       0,   0.5,     0).finished();
  Matrix Ractual = fs.tangent_space_projection(Y, Ydot);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, retract){
  cout << "\nTesting retract..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.r0 = 3;               // set r=3 for the Stiefel product manifold
  Fuses fs(crf, options);
  Matrix Y = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                             1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  Matrix Ydot = (Matrix(10, 3) << 1,0,0, 1,0,0, 2,0,0, 0,0,0, 0,0,1,
                                0,0,0, 0,0,1, 3,0,0, 0,1,0, 0,1,1).finished();

  // compare matrix values
  Matrix Rexpected = (Matrix(10, 3) << 1,                  0,                  0,
                       0.707106781186547,  0.707106781186547,                  0,
                                       1,                  0,                  0,
                                       1,                  0,                  0,
                                       0,                  0,                  1,
                                       1,                  0,                  0,
                                       0,                  0,                  1,
                                       1,                  0,                  0,
                                       0,  0.970142500145332, -0.242535625036333,
                                       0,  0.242535625036333,  0.970142500145332).finished();
  Matrix Ractual = fs.retract(Y, Ydot);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, round_solution){
  cout << "\nTesting round_solution..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  Fuses fs(crf);
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

  // compare matrix values
  Matrix Rexpected = (Matrix(7, 3) << 0,     0,     1,
                                      0,     0,     1,
                                      1,     0,     0,
                                      1,     0,     0,
                                      0,     1,     0,
                                      0,     0,     1,
                                      1,     0,     0).finished();
  Matrix Ractual = fs.round_solution(Y);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, round_solution2){
  cout << "\nTesting round_solution2..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
//  options.round = Sampling;
  Fuses fs(crf);
  Matrix Y = (Matrix(10, 4) << 0,   0.9,   0.1,   0.0,
                            -0.1,   0.2,   0.8,   0.0,
                             0.3,     0,     1,   0.0,
                               0,  0.01,   0.8,   0.0,
                               1,     0,     0,   0.0,
                               0,     2,     0,   0.0,
                               0,     0,  0.99,   0.0,
                               1,     0,     0,   0.0,
                               0,     1,     0,   0.0,
                               0,     0,     1,   0.0).finished();

  // compare matrix values
  //  Matrix Rexpected = (Matrix(7, 3) << 0,     1,     0,
  //                                      0,     0,     1,
  //                                      1,     0,     0,
  //                                      0,     0,     1,
  //                                      1,     0,     0,
  //                                      0,     1,     0,
  //                                      0,     0,     1).finished();

  // apparently the following improves the cost
  Matrix Rexpected = (Matrix(7, 3) <<
      0,     1,     0,
      0,     0,     1,
      0,     0,     1,
      0,     0,     1,
      1,     0,     0,
      0,     1,     0,
      0,     0,     1).finished();

  Matrix Ractual = fs.round_solution(Y);
  EXPECT(assert_equal(Rexpected, Ractual, tol));
}

/* ************************************************************************* */
TEST(testFuses, round_solution_sampling){
  cout << "\nTesting round_solution3..." << endl;
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  FUSESOpts options;
  options.round = Sampling;
  Fuses fs(crf);
  Matrix XX = (Matrix(7, 3) << 0,   0.9,   0.1,
                               0,   0.2,   0.8,
                             0.3,     0,   0.7,
                               0,   0.2,   0.8,
                               1,     0,     0,
                               0,     1,     0,
                            0.01,     0,  0.99).finished();

  std::default_random_engine generator;
  std::vector<std::discrete_distribution<int>> distributions =
      Fuses::GetProbabilitiesFromMatrix(XX);

  Matrix XP = Matrix::Zero(7,3);
  size_t n = 7;
  size_t k = 3;
  std::vector<size_t> labels; labels.resize(n); // n labels
  for(size_t sample = 0; sample < 10000; sample++){
    for(size_t i=0; i<n; i++){ // for each node
      size_t l = distributions.at(i)(generator);
      labels.at(i) = l;
    }
    XP += UtilsGM::GetMatrixFromLabels(labels, n, k);
  }
  XP = XP / 10000.0;
  EXPECT(assert_equal(XX, XP, 0.01)); // compare actual and empirical probabilities
}

/* ************************************************************************* */
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

/* ************************************************************************* */
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
TEST(testFuses, convertMatrixToVec){
  cout << "\nTesting convertMatrixToVec..." << endl;
  CRF crf;
  crf.nrNodes = 10;
  crf.nrClasses = 4;

  // prevent modifying nrClasses in fuses constructor
  UnaryFactor u1, u2, u3, u4;
  u1.node = 0;
  u1.label = 0;
  u1.weight = 0.1;
  crf.unaryFactors.push_back(u1);

  u2.node = 0;
  u2.label = 2;
  u2.weight = 0.8;
  crf.unaryFactors.push_back(u2);

  u3.node = 1;
  u3.label = 1;
  u3.weight = 1.0;
  crf.unaryFactors.push_back(u3);

  u4.node = 2;
  u4.label = 3;
  u4.weight = 0.9;
  crf.unaryFactors.push_back(u4);

  Fuses fs(crf);
  Matrix Y = (Matrix(10, 4) << 0,     1,     0,     0,
                               1,     0,     0,     0,
                               0,     0,     1,     0,
                               0,     1,     0,     0,
                               1,     0,     0,     0,
                               0,     0,     0,     1,
                               0,     0,     1,     0,
                               1,     0,     0,     0,
                               0,     1,     0,     0,
                               0,     0,     0,     1).finished();

  // expected and actual results
  vector<size_t> result_expected{1, 0, 2, 1, 0, 3, 2, 0, 1, 3};
  vector<size_t> result_actual = fs.convertMatrixToVec(Y);

  // compare results
  for(int i=0; i<10; ++i) {
    EXPECT(result_expected[i] == result_actual[i]);
  }

}

/* ************************************************************************* */
TEST(testFuses, solve1){
  cout << "\nTesting solve (This only test the solution not the status)..."
       << endl;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  Fuses fs1(crf1);
  fs1.solve();
  EXPECT(fs1.compare(train100Result, 0.0001));

  // Use data with 200 nodes
  SegmentationFrontEnd sfe2(train200Data);
  CRF crf2 = sfe2.getCRF();

  Fuses fs2(crf2);
  fs2.solve();
  EXPECT(fs2.compare(train200Result, 0.0001));

  // Use data with 500 nodes
  SegmentationFrontEnd sfe3(train500Data);
  CRF crf3 = sfe3.getCRF();

  Fuses fs3(crf3);
  fs3.solve();
  EXPECT(fs3.compare(train500Result, 0.0001));

  // Use data with 800 nodes
  SegmentationFrontEnd sfe4(train800Data);
  CRF crf4 = sfe4.getCRF();

  Fuses fs4(crf4);
  fs4.solve();
  EXPECT(fs4.compare(train800Result, 0.0001));

  // Use data with 1500 nodes
  SegmentationFrontEnd sfe5(train1500Data);
  CRF crf5 = sfe5.getCRF();

  Fuses fs5(crf5);
  fs5.solve();
  EXPECT(fs5.compare(train1500Result, 0.0001));
}

/* ************************************************************************* */
TEST(testFuses, solve2){
  cout << "\nTesting solve with Jacobi precon "
       << "(This only test the solution not the status)..."
       << endl;

  FUSESOpts options;
  options.precon = Jacobi;

  // Use data with 100 nodes
  SegmentationFrontEnd sfe1(train100Data);
  CRF crf1 = sfe1.getCRF();

  Fuses fs1(crf1, options);
  fs1.solve();
  EXPECT(fs1.compare(train100Result, 0.0001));

  // Use data with 200 nodes
  SegmentationFrontEnd sfe2(train200Data);
  CRF crf2 = sfe2.getCRF();

  Fuses fs2(crf2, options);
  fs2.solve();
  EXPECT(fs2.compare(train200Result, 0.0001));

  // Use data with 500 nodes
  SegmentationFrontEnd sfe3(train500Data);
  CRF crf3 = sfe3.getCRF();

  Fuses fs3(crf3, options);
  fs3.solve();
  EXPECT(fs3.compare(train500Result, 0.0001));

  // Use data with 800 nodes
  SegmentationFrontEnd sfe4(train800Data);
  CRF crf4 = sfe4.getCRF();

  Fuses fs4(crf4, options);
  fs4.solve();
  EXPECT(fs4.compare(train800Result, 0.0001));

  // Use data with 1500 nodes
  SegmentationFrontEnd sfe5(train1500Data);
  CRF crf5 = sfe5.getCRF();

  Fuses fs5(crf5, options);
  fs5.solve();
  EXPECT(fs5.compare(train1500Result, 0.0001));
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
