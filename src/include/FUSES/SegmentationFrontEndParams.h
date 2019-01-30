/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   SegmentationFrontEndParams.h
 * @brief  Class collecting the parameters for segmentation front end
 * @author Siyi Hu, Luca Carlone
 */

#ifndef SegmentationFrontEndParams_H_
#define SegmentationFrontEndParams_H_

#include <memory>
#include <unordered_map>
#include <boost/foreach.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <gtsam/slam/SmartFactorParams.h>

namespace CRFsegmentation {

///////////////////////////////////////////////////////////////////////////////////////
class SegmentationFrontEndParams
{
public:
  SegmentationFrontEndParams(
      // SUPERPIXEL PARAMS
      const int nr_sp = 2000,
      const int sp_iterations = 3,
      const float ratio = 0.075f,
      const int min_element_size = 50,
      const int nr_levels = 6,
      const int prior = 2,
      const int histogram_bins = 5,
      // SEGMENTATION PARAMS
      const double lambda1 = 0.2,
      const double lambda2 = 0.2,
      const double beta = 0.0001
  ) : num_sp_(nr_sp), sp_iterations_(sp_iterations), ratio_(ratio), min_element_size_(min_element_size),
  nr_levels_(nr_levels), prior_(prior), histogram_bins_(histogram_bins),
  lambda1_(lambda1), lambda2_(lambda2), beta_(beta) {}

  // superpixel params
  int num_sp_, sp_iterations_, min_element_size_, nr_levels_, prior_, histogram_bins_;
  float ratio_;

  // segmentation params
  double lambda1_, lambda2_, beta_;

  /* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
  // parse params YAML file
  bool parseYAML(const std::string& filepath){
    // make sure that each YAML file has %YAML:1.0 as first line
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cout << "Cannot open file in parseYAML: " << filepath << std::endl;
      throw std::runtime_error("parseYAML (Fuses): cannot open file (remember first line: %YAML:1.0)");
    }
    // SUPERPIXEL PARAMS
    fs["num_sp"] >> num_sp_;
    fs["sp_iterations"] >> sp_iterations_;
    fs["ratio"] >> ratio_;
    fs["min_element_size"] >> min_element_size_;
    fs["nr_levels"] >> nr_levels_;
    fs["prior"] >> prior_;
    fs["histogram_bins"] >> histogram_bins_;
    // SEGMENTATION PARAMS
    fs["lambda1"] >> lambda1_;
    fs["lambda2"] >> lambda2_;
    fs["beta"] >> beta_;

    fs.release();
    return true;
  }
  /* ------------------------------------------------------------------------------------- */
  bool equals(const SegmentationFrontEndParams& fp2, double tol = 1e-8) const{
    return
        // SUPERPIXEL PARAMS
        (num_sp_ == fp2.num_sp_) &&
        (sp_iterations_ == fp2.sp_iterations_) &&
        (fabs(ratio_ - fp2.ratio_) <= tol) &&
        (min_element_size_ == fp2.min_element_size_) &&
        (nr_levels_ == fp2.nr_levels_) &&
        (prior_ == fp2.prior_) &&
        (histogram_bins_ == fp2.histogram_bins_) &&
        // SEGMENTATION PARAMS
        (fabs(lambda1_ - fp2.lambda1_) <= tol) &&
        (fabs(lambda2_ - fp2.lambda2_) <= tol) &&
        (fabs(beta_ - fp2.beta_) <= tol);
  }
  /* ------------------------------------------------------------------------------------- */
  void print() const{
    std::cout << "$$$$$$$$$$$$$$$$$$$$$ SEGMENTATION FRONTEND PARAMETERS $$$$$$$$$$$$$$$$$$$$$" << std::endl;
    std::cout << "** Superpixel parameters **" << std::endl;
    std::cout << "num_sp_: " << num_sp_ << std::endl
        << "sp_iterations_: " << sp_iterations_ << std::endl
        << "ratio_: " << ratio_ << std::endl
        << "min_element_size_: " << min_element_size_ << std::endl
        << "nr_levels_: " << nr_levels_ << std::endl
        << "prior_: " << prior_ << std::endl
        << "histogram_bins_: " << histogram_bins_ << std::endl;
    std::cout << "** Segmentation parameters **" << std::endl
        << "lambda1_: " << lambda1_ << std::endl
        << "lambda2_: " << lambda2_ << std::endl
        << "beta_: " << beta_ << std::endl;
  }
};

} // namespace CRFsegmentation
#endif /* SegmentationFrontEndParams_H_ */


