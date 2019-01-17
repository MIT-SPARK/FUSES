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

#include "FUSES/OpenGMParser.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/UtilsOpenCV.h"
#include "FUSES/Timer.h"
#include "test_config.h"

using namespace gtsam;
using namespace std;
using namespace CRFsegmentation;
using namespace opengm;
using namespace myUtils::Timer;

static const string toyCRFfile = string(DATASET_PATH) + "/test.csv";
static const string negativeCRFfile = string(DATASET_PATH) + "/negativeCRF.csv";
//static const string train100Data = string(DATASET_PATH) + "/train100Data.csv";
//static const string train100Result = string(DATASET_PATH) + "/train100Result.csv";

/* ************************************************************************* */
// Data
static const double tol = 1e-7;

/* ************************************************************************* */
TEST(OpenGMParser, contructor_CRF){
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  // instantiate and get graph using openGM
  OpenGMParser fg(crf);
  CRF crf2 = fg.getCRF();

  //fg.saveModel("/home/luca/Desktop/toyCRFfile.hdf5");

  // check crf
  EXPECT(crf.equals(crf2, tol));
}

/* ************************************************************************* */
TEST(OpenGMParser, contructor_string){
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  // instantiate and get graph using openGM
  OpenGMParser fg(string(DATASET_PATH) + "/toyCRFfile.hdf5");
  CRF crf2 = fg.getCRF();

  // check crf
  EXPECT(crf.equals(crf2, tol));
}

/* ************************************************************************* */
TEST(OpenGMParser, computeLBP){
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  // instantiate and get graph using openGM
  bool verbose = false;
  OpenGMParser fg(string(DATASET_PATH) + "/toyCRFfile.hdf5");
  vector<size_t> labels;
  std::vector<double> iterations, values, times;
  fg.computeLBP(labels, iterations, values, times, verbose);

  // check that cost evaluation is consistent
  EXPECT_DOUBLES_EQUAL(crf.evaluate(labels), fg.evaluate(labels), tol);

  // check that cost evaluation matches iterations:
  EXPECT_DOUBLES_EQUAL(fg.evaluate(labels), values.back(), tol);

  // check that cost evaluation is consistent
  EXPECT(7 == fg.nrNodes());
  EXPECT(3 == fg.nrClasses());

  FUSES::UtilsOpenCV::PrintVector<size_t>(labels,"labels");
  FUSES::UtilsOpenCV::PrintVector<double>(values,"values");
  FUSES::UtilsOpenCV::PrintVector<double>(iterations,"iterations");
  FUSES::UtilsOpenCV::PrintVector<double>(times,"times");
  crf.print();
}

/* ************************************************************************* */
TEST(OpenGMParser, computeAE){
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  // instantiate and get graph using openGM
  bool verbose = false;
  OpenGMParser fg(string(DATASET_PATH) + "/toyCRFfile.hdf5");
  vector<size_t> labels;
  std::vector<double> iterations, values, times;
  fg.computeAE(labels, iterations, values, times, verbose);

  // check that cost evaluation is consistent
  EXPECT_DOUBLES_EQUAL(crf.evaluate(labels), fg.evaluate(labels), tol);

  // check that cost evaluation matches iterations:
  EXPECT_DOUBLES_EQUAL(fg.evaluate(labels), values.back(), tol);
}

/* ************************************************************************* */
TEST(OpenGMParser, computeTRWS){
  SegmentationFrontEnd sfe(toyCRFfile);
  CRF crf = sfe.getCRF();

  // instantiate and get graph using openGM
  bool verbose = false;
  OpenGMParser fg(string(DATASET_PATH) + "/toyCRFfile.hdf5");
  vector<size_t> labels;
  std::vector<double> iterations, values, times;
  fg.computeTRWS(labels, iterations, values, times, verbose);

  // check that cost evaluation is consistent
  EXPECT_DOUBLES_EQUAL(crf.evaluate(labels), fg.evaluate(labels), tol);

  // check that cost evaluation matches iterations:
  EXPECT_DOUBLES_EQUAL(fg.evaluate(labels), values.back(), tol);
}

/* ************************************************************************* */
TEST(OpenGMParser, negativeWeights){
  SegmentationFrontEnd sfe(negativeCRFfile);
  CRF crf = sfe.getCRF();

  // instantiate and get graph using openGM
  bool verbose = false;
  OpenGMParser fg(string(DATASET_PATH) + "/negativeCRFfile.hdf5");
  vector<size_t> labels;
  std::vector<double> iterations, values, times;
  fg.computeTRWS(labels, iterations, values, times, verbose);

  // check that cost evaluation is consistent
  EXPECT_DOUBLES_EQUAL(crf.evaluate(labels), fg.evaluate(labels), tol);

  // check that cost evaluation matches iterations:
  EXPECT_DOUBLES_EQUAL(fg.evaluate(labels), values.back(), tol);
  cout << fg.evaluate(labels) << endl;
  FUSES::UtilsOpenCV::PrintVector<size_t>(labels,"labels");
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
