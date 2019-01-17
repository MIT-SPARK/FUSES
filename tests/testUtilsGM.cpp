
/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testOpenGMParser.cpp
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
#include "FUSES/StiefelMixedProduct.h"
#include "FUSES/UtilsFuses.h"
#include "test_config.h"

using namespace gtsam;
using namespace std;

static const string toyCRFfile = string(DATASET_PATH) + "/test.csv";
static const string train100Data = string(DATASET_PATH) + "/train100Data.csv";
static const string train100Result = string(DATASET_PATH) + "/train100Result.csv";

/* ************************************************************************* */
// Data
static const double tol = 1e-7;

/* ************************************************************************* */
TEST(OpenGMParser, getLabelDiff){
  vector<size_t> labels;
  labels.push_back(0);
  labels.push_back(1);
  labels.push_back(3);
  labels.push_back(6);
  labels.push_back(1);

  vector<size_t> gtLabels{0, 1, 3, 6, 1};
  EXPECT(UtilsFuses::GetLabelDiff(labels,gtLabels) == 0);

  vector<size_t> gtLabels1{0, 1, 3, 4, 1};
  EXPECT(UtilsFuses::GetLabelDiff(labels,gtLabels1) == 1);
}

/* ************************************************************************* */
TEST(OpenGMParser, getLabelDiff_file){
  vector<size_t> labels;
  labels.push_back(0);
  labels.push_back(1);
  labels.push_back(3);
  labels.push_back(6);
  labels.push_back(1);

  EXPECT(UtilsFuses::GetLabelDiff(labels,string(DATASET_PATH) + "/test_labels.csv") == 0);
  EXPECT(UtilsFuses::GetLabelDiff(labels,string(DATASET_PATH) + "/test_labels1.csv") == 1);
}

/* ************************************************************************* */
TEST(OpenGMParser, reshape){
  Matrix M = (Matrix(2,4)
      << 0,  0.25,     -2,  -4,
      -0.25,    -1,     -3,  -5).finished();
  Matrix actualM = UtilsGM::Reshape(M,4,2);
  Matrix expectedM = (Matrix(4,2)
      << 0 ,  -2.0000,
      -0.2500  , -3.0000,
       0.2500 ,  -4.0000,
      -1.0000 ,  -5.0000).finished();
  EXPECT(assert_equal(expectedM, actualM, 1e-5));
}

/* ************************************************************************* */
TEST(OpenGMParser, GetLabelsFromMatrix){
  gtsam::Matrix L = gtsam::Matrix::Zero(5,3);
  L(0,0) = 1;
  L(1,1) = 1;
  L(2,2) = 1;
  L(3,0) = 1;
  L(4,1) = 1;

  vector<size_t> labelsFromMat = UtilsGM::GetLabelsFromMatrix(L,5,3);
  vector<size_t> labels{0, 1, 2, 0, 1};

  EXPECT(UtilsFuses::GetLabelDiff(labelsFromMat,labels) == 0);
}

/* ************************************************************************* */
TEST(OpenGMParser, GetMatrixFromLabels){
  gtsam::Matrix L = gtsam::Matrix::Zero(5,3);
  L(0,0) = 1;
  L(1,1) = 1;
  L(2,2) = 1;
  L(3,0) = 1;
  L(4,1) = 1;
  vector<size_t> labelsFromMat = UtilsGM::GetLabelsFromMatrix(L,5,3);

  gtsam::Matrix L_actual = UtilsGM::GetMatrixFromLabels(labelsFromMat,5,3);
  EXPECT(assert_equal(L, L_actual, 1e-5));

}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
