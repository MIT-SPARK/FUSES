/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * ------------------------------------------------------------------------- */

/**
 * @file   SegmentationFrontend.h
 * @brief  Creates CRF from RGB images
 * @author Siyi Hu, Luca Carlone
 */

#ifndef SegmentationFrontEnd_H_
#define SegmentationFrontEnd_H_

#include <ctype.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <map>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <gtsam/base/Lie.h>

#include "FUSES/CRF.h"
#include "FUSES/SegmentationFrontEndParams.h"
#include "FUSES/Timer.h"

namespace CRFsegmentation {

struct SuperPixel {
  cv::Vec3d avgColor;          // use double for precision
  std::pair<double, double> center;
  std::vector<cv::Vec3b> colors;    // BGR color stored in 3 bytes
  std::vector<std::pair<int, int>> pixels;
  int nrPixels = 0;
};

// parameters for generating superpixels with LSC algorithm
struct SPParam {
  int nr_sp = 2000;
  int sp_iterations = 3;

  //lsc
  float ratio = 0.075f;
  int min_element_size = 30;

  //seeds
  int nr_levels = 6;
  int prior = 2;
  int histogram_bins = 5;
};

// parameters for setting weights of the binary factors
struct SegParam {
  double lambda1 = 1;
  double lambda2 = 1;
  double beta = 0.001;
};

// load parameters from a yaml file
void loadParameters(SPParam& spParam, SegParam& segParam,
    const std::string& paramFile);

enum spAlgorithm { LSC = 1, SEEDS = 2, SLIC = 3, SLICO = 4, MSLIC = 5};

/*
 * Class describing data matrices for segmentation
 */
class SegmentationFrontEnd {
private:
  // CRF containing: nrNodes, nrClasses, unaryFactors, binaryFactors
  CRF crf_;

  // image ID
  int imgId_;

  // superpixels
  std::vector<SuperPixel> superPixels_;

  // superpixel labels
  std::vector<size_t> spLabels_;

public:
  // variance of color difference ||I_i - I_j|| between neighboring superpixels
  double colorVariance = 0;     // segParam.beta ~= 1/(2*colorVariance)

  /* ---------------------------------------------------------------------- */
  // constructors
  SegmentationFrontEnd(int imgId=0): imgId_(imgId) {};

  // constructor: get binary and unary factors from a csv file
  SegmentationFrontEnd(const std::string& filename, int imgId=0, int verbose=0);

  // constructor: get binary factors from images
  SegmentationFrontEnd(const cv::Mat& image, const SegParam& segParam,
      const SPParam& spParam=SPParam(), int spAlgorithm=SEEDS, int imgId=0,
      int verbose=0);

  // constructor: given a segmented image, get binary factors from images
  SegmentationFrontEnd(const cv::Mat& image, const SegParam& segParam,
      const cv::Mat& labels, int nrNodes, int imgId=0, int verbose=0);

  /* ---------------------------------------------------------------------- */
  // create super pixels using specified parameter and algorithm
  // the segmentation will be stored in labels
  // the contours will be stored in mask if verbose > 2
  void createSuperpixels(cv::Mat& image, cv::Mat& labels, cv::Mat& mask,
      const SPParam& spParam=SPParam(), int spAlgorithm=SEEDS,
      int verbose = 0);

  /* ---------------------------------------------------------------------- */
  // compute binary factors using connectivity matrix
  // and update colorVariance data member
  // Note: CRF only stores half of the binary terms to save time and memory
  // (i.e. if edge(i,j) is stored, then edge(j,i) will not be stored but
  // should have the same weight)
  void getBinaryFactors(const gtsam::Matrix& connectMatrix,
      const SegParam& segParam, int verbose=0);

  // in addition to neighboring SP, this add connections with other *close* SPs
  // (add connection when to centers are within radius measured in pixels)
  void getBinaryFactors(const gtsam::Matrix& connectMatrix, double radius,
      const SegParam& segParam, int verbose=0);

  // this is used when a colorDiffMat is given so that the average SP color in
  // the data member is ignored
  void getBinaryFactors(const gtsam::Matrix& connectMatrix,
      const cv::Mat& colorDiffMat, const SegParam& segParam, int verbose=0);

  /* ---------------------------------------------------------------------- */
  // read beta from average color of each superpixel for this image
  // Note: assume zero mean and beta = (2*variance)^-1
  void correctBinary(const std::vector<size_t>& labels,
      double attractiveWeight = 1.0, double repulsiveWeight = -1.0);

  /* ---------------------------------------------------------------------- */
  // read beta from average color of each superpixel for this image
  // Note: assume zero mean and beta = (2*variance)^-1
  double getBeta(const gtsam::Matrix& connectMatrix) const;

  // read beta from average color of each superpixel for this image
  // Note: assume zero mean and beta = (2*variance)^-1
  double getBeta(const gtsam::Matrix& connectMatrix,
      const cv::Mat& colorDiffMat) const;

  /* ---------------------------------------------------------------------- */
  // read ground-truth labels from label image
  // labels: input image storing label ID as uint8_t
  // label ID less than 0 is assumed to be in the last class (i.e. nrClasses-1)
  // labelVec: vector mapping label ID to class index (class = labelVec[label])
  void getGTLabelsFromID(const cv::Mat& labels, size_t nrClasses,
      const std::vector<size_t>& labelVec, int verbose=0);

  /* ---------------------------------------------------------------------- */
  // get unary factors from bonnet mask
  void getUnaryFactorsFromBonnet(const cv::Mat& bonnetLabel, size_t dstCols,
      size_t dstRows, bool constUnary=true, int verbose=0);

  // generate multiple unary factors based on probability
  void getUnaryFactorsFromBonnet(const cv::Mat& bonnetLabel,
      const cv::Mat& bonnetProb, size_t dstCols, size_t dstRows,
      int verbose=0);

  /* ---------------------------------------------------------------------- */
  // read unary factors from label image
  void getUnaryFactorsFromImage(const cv::Mat& labelImage,
      size_t pctClassified=20, size_t maxSamplingIter=100, bool constUnary=true,
      int verbose=0);

  /* ---------------------------------------------------------------------- */
  // helper function for getUnaryFactorsFromImage
  // to count the number of pixel level labels each super pixel has and update
  // crf_.nrClasses. Then it finds the ground-truth label for each superpixel.
  void getSPLabelsFromImage(const cv::Mat& inputLabel,
      std::vector<std::vector<size_t>>* totalLabelCount,
      std::vector<size_t>* spLabels, int verbose=0);

  /* ---------------------------------------------------------------------- */
  // helper function to add unary factors to crf_
  // if constUnary is true, this function will use a *majority-takes-all*
  // strategy such that one unary factor with weight 1 is created for the
  // specified node;
  // otherwise, if the dominant label in the given node has weight less than
  // 0.8, multiple unary factors with weight greater than 0.2 will be added.
  void addUnaryFactorsFromLabelCount(bool constUnary, int node,
      const std::vector<size_t>& labelCount);

  // The weightedLabelCount stores a vector of weights for each label, this
  // function adds the first k most probable labels of the specified node
  // to crf_
  // (less than k unary factors will be added if the weight is too small, but
  // at least one will be added to crf_)
  void addUnaryFactorsFromLabelCount(int node, size_t k, double minPerc,
      const std::vector<float>& weightedLabelCount);

  /* ---------------------------------------------------------------------- */
  // return reference to imagID
  const int& getID() const;

  /* ---------------------------------------------------------------------- */
  // return reference to spLabels
  const std::vector<size_t>& getSPLabels() const;

  /* ---------------------------------------------------------------------- */
  // return reference to crf_
  const CRF& getCRF() const;

  /* ---------------------------------------------------------------------- */
  // return reference to superPixels_
  const std::vector<SuperPixel>& getSP() const;

  /* ---------------------------------------------------------------------- */
  // display all params
  void print() const;

  /* ---------------------------------------------------------------------- */
  // display image with cluster mean
  void displayWithClusterMeans(const cv::Mat& inputImage) const;

  /* ---------------------------------------------------------------------- */
  // assert equality up to a tolerance
  bool equals(const SegmentationFrontEnd& sfe, double tol = 1e-9) const;

};

} // namespace CRFsegmentation

#endif /* SegmentationFrontend_H_ */
