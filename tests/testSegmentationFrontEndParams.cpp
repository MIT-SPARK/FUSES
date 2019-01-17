/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testSegmentationFrontEndParams.cpp
 * @brief  test SegmentationFrontEndParams
 * @author Siyi Hu, Luca Carlone
 */

#include <cstdlib>
#include <iostream>
#include <random>
#include <algorithm>
#include <CppUnitLite/TestHarness.h>

#include "FUSES/SegmentationFrontEndParams.h"
#include "test_config.h"

using namespace gtsam;
using namespace std;
using namespace CRFsegmentation;
using namespace cv;

static const double tol = 1e-7;
static const string filepath = string(DATASET_PATH) + "/fusesParameters_default.yaml";

/* ************************************************************************* */
TEST(testSegmentationFrontEndParams, FusesParseYAML) {
  // Test parseYAML
  SegmentationFrontEndParams fp;
  fp.parseYAML(filepath);

  // Check the parsed values!
  // Superpixel parms
  EXPECT_DOUBLES_EQUAL(2000, fp.num_sp_, tol);
  EXPECT_DOUBLES_EQUAL(10, fp.sp_iterations_, tol);
  EXPECT_DOUBLES_EQUAL(20, fp.min_element_size_, tol);
  EXPECT_DOUBLES_EQUAL(0.075, fp.ratio_, tol);
  EXPECT_DOUBLES_EQUAL(8, fp.nr_levels_, tol);
  EXPECT_DOUBLES_EQUAL(3, fp.prior_, tol);
  EXPECT_DOUBLES_EQUAL(5, fp.histogram_bins_, tol);
  // Superpixel parms
  EXPECT_DOUBLES_EQUAL(0.2, fp.lambda1_, tol);
  EXPECT_DOUBLES_EQUAL(0.2, fp.lambda2_, tol);
  EXPECT_DOUBLES_EQUAL(0.0001, fp.beta_, tol);

}

/* ************************************************************************* */
TEST(testSegmentationFrontEndParams, equals) {
  SegmentationFrontEndParams fp = SegmentationFrontEndParams();
  EXPECT(fp.equals(fp));

  SegmentationFrontEndParams fp2 = SegmentationFrontEndParams();
  fp2.beta_ += 1e-5; // small perturbation

  EXPECT(!fp.equals(fp2));
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
