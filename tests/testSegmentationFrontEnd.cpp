/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testSegmentationFrontEnd.cpp
 * @brief  unit test for image segmentation front end
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
#include "FUSES/SegmentationFrontEnd.h"
#include "test_config.h"

using namespace gtsam;
using namespace std;
using namespace CRFsegmentation;

static const string toyCRFfile = string(DATASET_PATH) + "/test.csv";
static const string largeCRFfile = string(DATASET_PATH) + "/train100Data.csv";
static const string trainImageName = string(DATASET_PATH) + "/trainImage.jpg";
static const string trainLabelName = string(DATASET_PATH) + "/trainLabel.png";
static const string rgbImageName = string(DATASET_PATH) + "/testRGBImage.png";
static const string bonnetImageName = string(DATASET_PATH) + "/testBonnetImage.png";
static const string param_defaultFile = string(DATASET_PATH) + "/fusesParameters_default.yaml";


/* ************************************************************************* */
// Data
static const double tol = 1e-7;
cv::Mat trainImage = cv::imread(trainImageName);
cv::Mat trainLabel = cv::imread(trainLabelName);
cv::Mat rgbImage = cv::imread(rgbImageName);
cv::Mat bonnetImage = cv::imread(bonnetImageName);

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, contructor1){
  cout << "\nTesting constructor1..." << endl;
  // default constructor
  SegmentationFrontEnd sfe(3);
  EXPECT(sfe.getID() == 3);
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, contructor2){
  cout << "\nTesting constructor2..." << endl;
  // instantiate front end given CRF data:
  SegmentationFrontEnd sfe(toyCRFfile, 3);
  CRF crf = sfe.getCRF();

  // check imgID
  EXPECT(sfe.getID() == 3);

  // check nrClasses
  EXPECT(crf.nrClasses == 3);

  // check unary and binary factors
  EXPECT(crf.unaryFactors.size() == 5);
  EXPECT(crf.binaryFactors.size() == 7);

  EXPECT(crf.unaryFactors[0].node == 0);
  EXPECT(crf.unaryFactors[0].label == 0);

  EXPECT(crf.unaryFactors[1].node == 1);
  EXPECT(crf.unaryFactors[1].label == 0);

  EXPECT(crf.unaryFactors[2].node == 2);
  EXPECT(crf.unaryFactors[2].label == 1);

  EXPECT(crf.unaryFactors[3].node == 3);
  EXPECT(crf.unaryFactors[3].label == 0);

  EXPECT(crf.unaryFactors[4].node == 4);
  EXPECT(crf.unaryFactors[4].label == 2);

  EXPECT(crf.binaryFactors[0].firstNode == 0);
  EXPECT(crf.binaryFactors[0].secondNode == 1);
  EXPECT_DOUBLES_EQUAL(crf.binaryFactors[0].weight, 0.25, tol);

  EXPECT(crf.binaryFactors[1].firstNode == 0);
  EXPECT(crf.binaryFactors[1].secondNode == 3);
  EXPECT_DOUBLES_EQUAL(crf.binaryFactors[1].weight, 0.15, tol);

  EXPECT(crf.binaryFactors[2].firstNode == 1);
  EXPECT(crf.binaryFactors[2].secondNode == 2);
  EXPECT_DOUBLES_EQUAL(crf.binaryFactors[2].weight, 0.225, tol);

  EXPECT(crf.binaryFactors[3].firstNode == 1);
  EXPECT(crf.binaryFactors[3].secondNode == 3);
  EXPECT_DOUBLES_EQUAL(crf.binaryFactors[3].weight, 0.2, tol);

  EXPECT(crf.binaryFactors[4].firstNode == 2);
  EXPECT(crf.binaryFactors[4].secondNode == 3);
  EXPECT_DOUBLES_EQUAL(crf.binaryFactors[4].weight, 0.175, tol);

  EXPECT(crf.binaryFactors[5].firstNode == 2);
  EXPECT(crf.binaryFactors[5].secondNode == 4);
  EXPECT_DOUBLES_EQUAL(crf.binaryFactors[5].weight, 0.1, tol);

  EXPECT(crf.binaryFactors[6].firstNode == 3);
  EXPECT(crf.binaryFactors[6].secondNode == 4);
  EXPECT_DOUBLES_EQUAL(crf.binaryFactors[6].weight, 0.05, tol);

}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, contructor3){
  cout << "\nTesting constructor3..." << endl;
  // instantiate front end given CRF data:
  SegmentationFrontEnd sfe(largeCRFfile, 3);
  CRF crf = sfe.getCRF();

  // check the number of unary and binary factors
  EXPECT(crf.unaryFactors.size() == 28);
  EXPECT(crf.binaryFactors.size() == 238);

  // check some components in crf_.unaryFactors
  for(UnaryFactor& unaryFactor : crf.unaryFactors) {
    if(unaryFactor.node == 54) {
      EXPECT_DOUBLES_EQUAL(unaryFactor.label, 0, tol);
    } else if(unaryFactor.node == 37) {
      EXPECT_DOUBLES_EQUAL(unaryFactor.label, 3, tol);
    } else if(unaryFactor.node == 26) {
      EXPECT_DOUBLES_EQUAL(unaryFactor.label, 1, tol);
    } else if(unaryFactor.node == 90) {
      EXPECT_DOUBLES_EQUAL(unaryFactor.label, 0, tol);
    }
  }

  // check some components in crf_.binaryFactors
  for(BinaryFactor& binaryFactor : crf.binaryFactors) {
    if(binaryFactor.firstNode == 2 && binaryFactor.secondNode == 9) {
      EXPECT_DOUBLES_EQUAL(binaryFactor.weight, 0.097313, tol);
    } else if(binaryFactor.firstNode == 26 && binaryFactor.secondNode == 28) {
      EXPECT_DOUBLES_EQUAL(binaryFactor.weight, 0.051398, tol);
    } else if(binaryFactor.firstNode == 68 && binaryFactor.secondNode == 69) {
      EXPECT_DOUBLES_EQUAL(binaryFactor.weight, 0.075769, tol);
    } else if(binaryFactor.firstNode == 83 && binaryFactor.secondNode == 84) {
      EXPECT_DOUBLES_EQUAL(binaryFactor.weight, 0.07237, tol);
    } else if(binaryFactor.firstNode == 15) {
      EXPECT(binaryFactor.secondNode != 70);
    } else if(binaryFactor.firstNode == 33) {
      EXPECT(binaryFactor.secondNode != 54);
    }
  }
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, contructor4){
  cout << "\nTesting constructor4 (This also tests getBinaryFactors function)..." << endl;
  // instantiate front end with image and segmentation parameters:
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, param_defaultFile);
  SegmentationFrontEnd sfe(trainImage, segParam, spParam, LSC);
  CRF crf = sfe.getCRF();

  // check imgID
  EXPECT(sfe.getID() == 0);

  // check the number of superpixel nodes
  EXPECT(crf.nrNodes < 2500); // this number varies, but should be close to 2000
  EXPECT(crf.nrNodes > 1500);
  EXPECT(crf.nrNodes == sfe.getSP().size());

  // check each SuperPixel struct in superPixels_ vector has the right number of colors_ and pixels_
  auto superPixels = sfe.getSP();
  for(auto &SP : superPixels) {
    EXPECT(SP.nrPixels != 0); // each SP in superPixels_ vector should not be empty
    EXPECT(SP.nrPixels == SP.colors.size());
    EXPECT(SP.nrPixels == SP.pixels.size());
  }

  // check sum of nrPixels_ is the same as the number of pixels in the image
  int totalPixels = 0;
  for(auto &SP : superPixels) {
    totalPixels += SP.nrPixels;
  }
  EXPECT(totalPixels == 480*854);

  // check superpixel centers are within bounds
  for(auto &SP : superPixels) {
    EXPECT(SP.center.first > 0);
    EXPECT(SP.center.first < 480);
    EXPECT(SP.center.second > 0);
    EXPECT(SP.center.second < 854);
  }

  // check the number of binary factors
  EXPECT(crf.binaryFactors.size() < 4*crf.nrNodes);
  EXPECT(crf.binaryFactors.size() > 2*crf.nrNodes);

  // check each binaryFactor
  for(auto& binaryFactor : crf.binaryFactors) {
    EXPECT(binaryFactor.firstNode >= 0);
    EXPECT(binaryFactor.firstNode < crf.nrNodes);
    EXPECT(binaryFactor.secondNode >= 0);
    EXPECT(binaryFactor.secondNode < crf.nrNodes);
    EXPECT(binaryFactor.weight >= segParam.lambda1);
    EXPECT(binaryFactor.weight < segParam.lambda1 + segParam.lambda2);
  }

}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, contructor5){
  cout << "\nTesting constructor5 (This also tests getBinaryFactors function)..." << endl;
  // instantiate front end with image and segmentation parameters:
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, param_defaultFile);
  SegmentationFrontEnd sfe(trainImage, segParam, spParam, SEEDS, 10);
  CRF crf = sfe.getCRF();

  // check imgID
  EXPECT(sfe.getID() == 10);

  // check the number of superpixel nodes
  EXPECT(crf.nrNodes < 2500); // this number varies, but should be close to 2000
  EXPECT(crf.nrNodes > 1500);
  auto superPixels = sfe.getSP();
  EXPECT(crf.nrNodes == superPixels.size());

  // check each SuperPixel struct in superPixels_ vector has the right number of colors_ and pixels_
  for(auto &SP : superPixels) {
    EXPECT(SP.nrPixels != 0); // each SP in superPixels_ vector should not be empty
    EXPECT(SP.nrPixels == SP.colors.size());
    EXPECT(SP.nrPixels == SP.pixels.size());
  }

  // check sum of nrPixels_ is the same as the number of pixels in the image
  int totalPixels = 0;
  for(auto &SP : superPixels) {
    totalPixels += SP.nrPixels;
  }
  EXPECT(totalPixels == 480*854);

  // check superpixel centers are within bounds
  for(auto &SP : superPixels) {
    EXPECT(SP.center.first > 0);
    EXPECT(SP.center.first < 480);
    EXPECT(SP.center.second > 0);
    EXPECT(SP.center.second < 854);
  }

  // check the number of binary factors
  EXPECT(crf.binaryFactors.size() < 4*crf.nrNodes);
  EXPECT(crf.binaryFactors.size() > 2*crf.nrNodes);

  // check each binaryFactor
  for(auto& binaryFactor : crf.binaryFactors) {
    EXPECT(binaryFactor.firstNode >= 0);
    EXPECT(binaryFactor.firstNode < crf.nrNodes);
    EXPECT(binaryFactor.secondNode >= 0);
    EXPECT(binaryFactor.secondNode < crf.nrNodes);
    EXPECT(binaryFactor.weight >= segParam.lambda1);
    EXPECT(binaryFactor.weight < segParam.lambda1 + segParam.lambda2);
  }
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, getSPLabelsFromImage){
  cout << "\nTesting getSPLabelCount..." << endl;
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, param_defaultFile);
  SegmentationFrontEnd sfe(trainImage, segParam, spParam);

  vector<vector<size_t>> totalLabelCount;
  vector<size_t> spLabels;
  sfe.getSPLabelsFromImage(trainLabel, &totalLabelCount, &spLabels);
  CRF crf = sfe.getCRF();

  // all pixels are counted
  int totalPixels = 0;
  for(int i=0; i<crf.nrNodes; ++i) {
    for(int j=0; j<totalLabelCount[i].size(); ++j) {
      totalPixels += totalLabelCount[i][j];
    }
  }
  EXPECT(totalPixels == 480*854);

  // should create 5 classes
  EXPECT(crf.nrClasses == 5);

  // all superpixel are included
  EXPECT(totalLabelCount.size() == sfe.getSP().size());
  EXPECT(spLabels.size() == sfe.getSP().size());
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, getUnaryFactorsFromBonnet){
  cout << "\nTesting getUnaryFactorsFromBonnet..." << endl;
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, param_defaultFile);
  SegmentationFrontEnd sfe(rgbImage, segParam, spParam);

  // Case 1: unary factors with constant weight
  bool constUnary = true;
  sfe.getUnaryFactorsFromBonnet(bonnetImage, rgbImage.cols, rgbImage.rows,
      constUnary);
  CRF crf = sfe.getCRF();

  // all superpixels should be labeled
  EXPECT(crf.unaryFactors.size() == crf.nrNodes);

  // bonnet labels should be between 0 and 19
  for(auto &unaryFactor : crf.unaryFactors) {
    EXPECT(unaryFactor.node >= 0);
    EXPECT(unaryFactor.node < crf.nrNodes);
    EXPECT(unaryFactor.label >= 0);
    EXPECT(unaryFactor.label <= 19);
    EXPECT_DOUBLES_EQUAL(unaryFactor.weight, 1, tol);
  }

  // Case 2: unary factors with non-constant weight
  constUnary = false;
  sfe.getUnaryFactorsFromBonnet(bonnetImage, rgbImage.cols, rgbImage.rows,
      constUnary);
  crf = sfe.getCRF();

  // unary factors should be larger than nrNodes
  // (although some might be unlabeled)
  EXPECT(crf.unaryFactors.size() > crf.nrNodes);

  // 1) bonnet labels should be between 0 and 19
  // 2) at least one of the unary factors should have weight less than 1
  // 3) sum of weights should be less than nrNodes
  bool weightLessThanOne = false;
  double totalWeight = 0;
  for(auto &unaryFactor : crf.unaryFactors) {
    EXPECT(unaryFactor.node >= 0);
    EXPECT(unaryFactor.node < crf.nrNodes);
    EXPECT(unaryFactor.label >= 0);
    EXPECT(unaryFactor.label <= 19);
    EXPECT(unaryFactor.weight > 0.2);
    EXPECT(unaryFactor.weight <= 1);
    if(1 - unaryFactor.weight > tol) {
      weightLessThanOne = true;
    }
    totalWeight += unaryFactor.weight;
  }
  EXPECT(weightLessThanOne == true);
  EXPECT(totalWeight < crf.nrNodes);
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, getUnaryFactorsFromImage){
  cout << "\nTesting getUnaryFactorsFromImage..." << endl;
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, param_defaultFile);
  SegmentationFrontEnd sfe(trainImage, segParam, spParam);

  // Case 1: unary factors with constant weight
  bool constUnary = true;
  sfe.getUnaryFactorsFromImage(trainLabel, 20, 100, constUnary);
  CRF crf = sfe.getCRF();

  // 20% superpixels should be labeled
  EXPECT(crf.unaryFactors.size() == crf.nrNodes/5);

  // train labels should be between 0 and 4
  for(auto &unaryFactor : crf.unaryFactors) {
    EXPECT(unaryFactor.node >= 0);
    EXPECT(unaryFactor.node < crf.nrNodes);
    EXPECT(unaryFactor.label >= 0);
    EXPECT(unaryFactor.label <= 19);
    EXPECT_DOUBLES_EQUAL(unaryFactor.weight, 1, tol);
  }

  // Case 2: unary factors with non-constant weight
  constUnary = false;
  sfe.getUnaryFactorsFromImage(trainLabel, 20, 100, constUnary);
  crf = sfe.getCRF();

  // 20% superpixels should be labeled
  EXPECT(crf.unaryFactors.size() > crf.nrNodes/5);

  // 1) train labels should be between 0 and 4
  // 2) at least one of the unary factors should have weight less than 1
  // 3) sum of weights should be less than nrNodes
  bool weightLessThanOne = false;
  double totalWeight = 0;
  for(auto &unaryFactor : crf.unaryFactors) {
    EXPECT(unaryFactor.node >= 0);
    EXPECT(unaryFactor.node < crf.nrNodes);
    EXPECT(unaryFactor.label >= 0);
    EXPECT(unaryFactor.label <= 4);
    EXPECT(unaryFactor.weight > 0.2);
    EXPECT(unaryFactor.weight <= 1);
    if(1 - unaryFactor.weight > tol) {
      weightLessThanOne = true;
    }
    totalWeight += unaryFactor.weight;
  }
  EXPECT(weightLessThanOne == true);
  EXPECT(totalWeight < crf.nrNodes);
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, loadParameters) {
  cout << "\nTesting loadParameters..." << endl;
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, param_defaultFile);

  // check superpixel parameters
  EXPECT_DOUBLES_EQUAL(spParam.nr_sp, 2000, tol);
  EXPECT_DOUBLES_EQUAL(spParam.sp_iterations, 10, tol);
  EXPECT_DOUBLES_EQUAL(spParam.ratio, 0.075, tol);
  EXPECT_DOUBLES_EQUAL(spParam.min_element_size, 20, tol);
  EXPECT_DOUBLES_EQUAL(spParam.nr_levels, 8, tol);
  EXPECT_DOUBLES_EQUAL(spParam.prior, 3, tol);
  EXPECT_DOUBLES_EQUAL(spParam.histogram_bins, 5, tol);

  // check segmentation parameters
  EXPECT_DOUBLES_EQUAL(segParam.lambda1, 0.2, tol);
  EXPECT_DOUBLES_EQUAL(segParam.lambda2, 0.2, tol);
  EXPECT_DOUBLES_EQUAL(segParam.beta, 0.0001, tol);
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, unaryFactor_equals) {
  cout << "\nTesting unaryFactor_equals..." << endl;
  UnaryFactor uf1, uf2, uf3, uf4, uf5;
  uf1.node = 33;
  uf1.label = 2;

  uf2.node = 33;
  uf2.label = 2;

  uf3.node = 12;
  uf3.label = 2;

  uf4.node = 33;
  uf4.label = 5;

  uf5.node = 17;
  uf5.label = 1;
  EXPECT(uf1.equals(uf1));
  EXPECT(uf1.equals(uf2));
  EXPECT(!uf1.equals(uf3));
  EXPECT(!uf1.equals(uf4));
  EXPECT(!uf1.equals(uf5));
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, binaryFactor_equals) {
  cout << "\nTesting binaryFactor_equals..." << endl;
  BinaryFactor bf1, bf2, bf3, bf4;
  bf1.firstNode = 1;
  bf1.secondNode = 2;
  bf1.weight = 0.05;

  bf2.firstNode = 2;
  bf2.secondNode = 1;
  bf2.weight = 0.05;

  bf3.firstNode = 1;
  bf3.secondNode = 2;
  bf3.weight = 0.055;

  bf4.firstNode = 3;
  bf4.secondNode = 2;
  bf4.weight = 0.05;

  EXPECT(bf1.equals(bf1, tol));
  EXPECT(bf1.equals(bf2, tol));
  EXPECT(!bf1.equals(bf3, tol));
  EXPECT(!bf1.equals(bf4, tol));
}

/* ************************************************************************* */
TEST(testSegmentationFrontEnd, equals) {
  cout << "\nTesting equals..." << endl;
  SegParam segParam;
  SPParam spParam;

  // initialize sfe1 and sfe2 in the same way
  SegmentationFrontEnd sfe1(trainImage, segParam, spParam, SEEDS);
  SegmentationFrontEnd sfe2(trainImage, segParam, spParam, SEEDS);
  EXPECT(sfe1.equals(sfe2, tol));

  // add unary terms
  sfe1.getUnaryFactorsFromImage(trainLabel, 20);
  sfe2.getUnaryFactorsFromImage(trainLabel, 20);
  EXPECT(sfe1.equals(sfe2, tol));

  // different initialization
  SegmentationFrontEnd sfe3(trainImage, segParam, spParam, SLIC);
  sfe3.getUnaryFactorsFromImage(trainLabel, 20);
  EXPECT(!sfe1.equals(sfe3, tol));
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
