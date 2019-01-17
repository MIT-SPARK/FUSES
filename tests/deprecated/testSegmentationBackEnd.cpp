/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testSegmentationBackend.cpp
 * @brief  unit test for image segmentation back end
 * @author Siyi Hu, Luca Carlone
 */

#include <vector>
#include <string>
#include <cctype>
#include <iostream>
#include <fstream>
#include <cmath>
#include <gtsam/base/Lie.h>
#include <CppUnitLite/TestHarness.h>

#include "../src/SegmentationBackEnd.h"
#include "../src/SegmentationFrontEnd.h"
#include "test_config.h"

using namespace gtsam;
using namespace std;
using namespace CRFsegmentation;

static const string trainImage = string(DATASET_PATH) + "/trainImage.jpg";
static const string trainLabel = string(DATASET_PATH) + "/trainLabel.png";
static const string param_defaultFile = string(DATASET_PATH) + "/fusesParameters_default.yaml";

/* ************************************************************************* */
// Data
static const double tol = 1e-7;

/* ************************************************************************* */
TEST(testSegmentationBackEnd, contructor1){
  cout << "\nTesting constructor1..." << endl;
  // default constructor
  SegmentationBackEnd sbe(5);
  EXPECT(sbe.imgId_ == 5);
}

/* ************************************************************************* */
TEST(testSegmentationFrontend, contructor2){
  cout << "\nTesting constructor2..." << endl;
  // instantiate front end with image and segmentation parameters:
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, param_defaultFile);
  SegmentationFrontEnd sfe(trainImage, segParam, spParam, LSC);

  // constructor
  Matrix labels = Matrix::Random(sfe.crf_.nrNodes, sfe.crf_.nrClasses);
  SegmentationBackEnd sbe(sfe.superPixels_, labels);

  // check imgID
  EXPECT(sbe.imgId_ == 0);

  // check superPixels size: this number varies, but should be close to 2000
  EXPECT(sbe.superPixels_ -> size() < 2500);
  EXPECT(sbe.superPixels_ -> size() > 1500);

  // check sum of nrPixels is the same as the number of pixels in the image
  int totalPixels = 0;
  for(auto& SP : *sbe.superPixels_) {
    totalPixels += SP.nrPixels;
  }
  EXPECT(totalPixels == 480*854);

  // check labels_
  Matrix copy = Matrix(labels);
  EXPECT(assert_equal(copy, *(sbe.labels_), tol));
}

/* ************************************************************************* */
TEST(testSegmentationFrontend, setSP){
  cout << "\nTesting setSP..." << endl;
  // instantiate front end with image and segmentation parameters:
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, param_defaultFile);
  SegmentationFrontEnd sfe(trainImage, segParam, spParam, LSC);

  // constructor
  Matrix labels = Matrix::Random(sfe.crf_.nrNodes, sfe.crf_.nrClasses);
  SegmentationBackEnd sbe(sfe.superPixels_, labels);

  // reset superPixels_
  vector<SuperPixel> superPxielsReduced(100);
  copy(sfe.superPixels_.begin(), sfe.superPixels_.begin() + 100,
      superPxielsReduced.begin());
  sbe.setSP(superPxielsReduced);

  // check superPixels size
  EXPECT(sbe.superPixels_ -> size() == 100);

  // check sum of nrPixels is reduced
  int totalPixels = 0;
  for(auto& SP : *sbe.superPixels_) {
    totalPixels += SP.nrPixels;
  }
  EXPECT(totalPixels < 480*854);
  EXPECT(totalPixels > 100);
}

/* ************************************************************************* */
TEST(testSegmentationFrontend, setLabels){
  cout << "\nTesting setLabels..." << endl;
  // constructor
  Matrix labels = Matrix::Random(100, 3);
  vector<SuperPixel> superPixel;
  SegmentationBackEnd sbe(superPixel, labels);

  // reset labels_
  Matrix labelsNew = Matrix::Random(200, 5);
  sbe.setLabels(labelsNew);

  // check labels_
  Matrix copy = Matrix(labelsNew);
  EXPECT(assert_equal(copy, *(sbe.labels_), tol));
}

/* ************************************************************************* */
TEST(testSegmentationFrontend, compareLabels){
  cout << "\nTesting compareLabels..." << endl;
  // constructor
  Matrix labels = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                 1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  vector<SuperPixel> superPixel;
  SegmentationBackEnd sbe(superPixel, labels);

  // compare labels
  Matrix labelsGT = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 0,1,0, 0,0,1,
                                 1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,1,0).finished();
  EXPECT_DOUBLES_EQUAL(2, sbe.compareLabels(labelsGT), tol);

  vector<int> labelsGTVec{0, 0, 2, 0, 1, 0, 2, 1, 1, 0};
  EXPECT_DOUBLES_EQUAL(sqrt(10), sbe.compareLabels(labelsGTVec), tol);
}

/* ************************************************************************* */
TEST(testSegmentationFrontend, getLabelDiff){
  cout << "\nTesting getLabelDiff..." << endl;
  // constructor
  Matrix labels = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 1,0,0, 0,0,1,
                                 1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1).finished();
  vector<SuperPixel> superPixel;
  SegmentationBackEnd sbe(superPixel, labels);

  // compare labels
  Matrix labelsGT = (Matrix(10, 3) << 1,0,0, 0,1,0, 1,0,0, 0,1,0, 0,0,1,
                                 1,0,0, 0,0,1, 1,0,0, 0,1,0, 0,1,0).finished();
  EXPECT(sbe.getLabelDiff(labelsGT) == 2);

  vector<int> labelsGTVec{0, 0, 2, 0, 1, 0, 2, 1, 1, 0};
  EXPECT(sbe.getLabelDiff(labelsGTVec) == 5);
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
