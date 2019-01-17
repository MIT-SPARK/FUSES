/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testCRF.cpp
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
#include "test_config.h"

using namespace gtsam;
using namespace std;
using namespace CRFsegmentation;
using namespace FUSES;

static const string toyCRFfile = string(DATASET_PATH) + "/test.csv";
static const string toyCRFfileBinary = string(DATASET_PATH) + "/test1.csv";
static const string toyCRFfile2 = string(DATASET_PATH) + "/test2.csv";
static const string toyCRFfile3 = string(DATASET_PATH) + "/test3.csv";
static const string toyCRFfile4 = string(DATASET_PATH) + "/test4.csv";
static const string train100Data = string(DATASET_PATH) + "/train100Data.csv";
static const string train100Result = string(DATASET_PATH) + "/train100Result.csv";

/* ************************************************************************* */
// Data
static const double tol = 1e-7;

/* ************************************************************************* */
TEST(testCRF, contructor){
  // TODO LC: there should be no frontend here
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

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
TEST(testCRF, evaluateMRF){
  // TODO LC: there should be no frontend here
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  vector<size_t> labels1;
  labels1.push_back(0);
  labels1.push_back(0);
  labels1.push_back(0);
  labels1.push_back(0);
  labels1.push_back(0);
  labels1.push_back(0);
  labels1.push_back(0);
  double cost1 = crf.evaluateMRF(labels1);
  double expectedCost1 = 2.0; // 2 unary terms violated
  EXPECT_DOUBLES_EQUAL(expectedCost1, cost1, tol);

  vector<size_t> labels2;
  labels2.push_back(0);
  labels2.push_back(0);
  labels2.push_back(1);
  labels2.push_back(0);
  labels2.push_back(2);
  labels2.push_back(0);
  labels2.push_back(0);
  double cost2 = crf.evaluateMRF(labels2);
  // CRF describes an undirected graph, hence edges (i,j) implies presence of edge (j,1)
  // For simplicity the CRF structure only reports one edge, with the understanding that the
  // weight should be double counted
  double expectedCost2 = 2*(0.225 + 0.175 + 0.1 + 0.05); // 4 binary terms violated
  EXPECT_DOUBLES_EQUAL(expectedCost2, cost2, tol);

  vector<size_t> labels3;
  labels3.push_back(0);
  labels3.push_back(1);
  labels3.push_back(3);
  labels3.push_back(4);
  labels3.push_back(5);
  labels3.push_back(0);
  labels3.push_back(0);
  double cost3 = crf.evaluateMRF(labels3);
  // CRF describes an undirected graph, hence edges (i,j) implies presence of edge (j,1)
  // For simplicity the CRF structure only reports one edge, with the understanding that the
  // weight should be double counted
  double expectedCost3 = 2*(0.25 + 0.15 + 0.225 + 0.2 + 0.175 + 0.1 + 0.05) + 4; // all binary terms violated + some unary
  EXPECT_DOUBLES_EQUAL(expectedCost3, cost3, tol);
}

/* ************************************************************************* */
TEST(testCRF, equal){
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  CRF crf2 = crf;
  EXPECT(crf.equals(crf2, tol));

  CRF crf3 = crf;
  crf3.nrNodes = 3;
  EXPECT(!crf.equals(crf3, tol));

  CRF crf4 = crf;
  crf4.nrClasses = 1;
  EXPECT(!crf.equals(crf4, tol));

  CRF crf5 = crf;
  crf5.unaryFactors[1].node = 3;
  EXPECT(!crf.equals(crf5, tol));

  CRF crf6 = crf;
  crf6.unaryFactors[1].label = 3;
  EXPECT(!crf.equals(crf6, tol));

  CRF crf7 = crf;
  crf7.binaryFactors[1].firstNode = 3;
  EXPECT(!crf.equals(crf7, tol));

  CRF crf8 = crf;
  crf8.binaryFactors[2].secondNode = 3;
  EXPECT(!crf.equals(crf8, tol));

  CRF crf9 = crf;
  crf9.binaryFactors[3].weight = 3.0;
  EXPECT(!crf.equals(crf9, tol));
}

/* ************************************************************************* */
TEST(testCRF, reduceNrClassesCRF){
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();
  //  crf.print();
  //  EXPECT(false);
  crf.reduceNrClassesCRF(2); // reduce to binary segmentation

  SegmentationFrontEnd sfeBinary(toyCRFfileBinary);
  CRF crfBinary = sfeBinary.getCRF();
  EXPECT(crf.equals(crfBinary, tol));
}

/* ************************************************************************* */
TEST(testCRF, reduceNrClassesCRF2){
  SegmentationFrontEnd sfe(toyCRFfile2);
  CRF crf = sfe.getCRF();
  crf.reduceNrClassesCRF(4); // reduce to 4 classes

  SegmentationFrontEnd sfe2(toyCRFfile3);
  CRF crf2 = sfe2.getCRF();
  EXPECT(crf.equals(crf2, tol));
}

/* ************************************************************************* */
TEST(testCRF, reduceNrClassesCRF3){
  SegmentationFrontEnd sfe(toyCRFfile2);
  CRF crf = sfe.getCRF();
  crf.reduceNrClassesCRF2(3); // reduce to 4 classes

  SegmentationFrontEnd sfe2(toyCRFfile4);
  CRF crf2 = sfe2.getCRF();
  EXPECT(crf.equals(crf2, tol));
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
