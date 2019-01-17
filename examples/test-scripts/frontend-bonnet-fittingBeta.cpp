/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   fuses-bonnet-fittingBeta.cpp
 * @brief  log variance of color difference for all images in a specified
 *         Cityscapes dataset folder
 * @author Siyi Hu, Luca Carlone
 */

#ifdef USE_BONNET
#include "FUSES/OpenGMParser.h"
#include "FUSES/CityscapesParser.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/Timer.h"
#include "FUSES/UtilsFuses.h"
#include "bonnet.hpp"
#include "IDToLabelConst.h"

using namespace std;
using namespace CRFsegmentation;
using namespace opengm;
using namespace myUtils::Timer;

int main(int argc, char **argv) {

  bool verbose = false;
  string outputFile = "colorVariances.csv";

  if (argc < 5) {
    cout << "Usage: " << argv[0] << " [path/to/dataset/directory] [nameOfDataset] [path/to/frozen/model] [path/to/param/file]" << endl;
    exit(1);
  }

  // cityscapes dataset parser
  CityscapesParser dataset(argv[1], argv[2], string(argv[3]));
  size_t nrImages = dataset.nrImages();

  // front end parameters
  int spAlgorithm = LSC;
  SPParam spParam;
  SegParam segParam;
  string paramFile = argv[4];
  loadParameters(spParam, segParam, paramFile);

  // network stuff
  cv::Mat mask_argmax, prob_max;
  bonnet::retCode status;
  std::unique_ptr<bonnet::Bonnet> net;
  string device = "/gpu:0";
  string backend = "trt";
  string path = argv[3];

  // initialize network
  try {
    net = std::unique_ptr<bonnet::Bonnet>(
        new bonnet::Bonnet(path, backend, device, verbose));
  } catch (const std::invalid_argument &e) {
    std::cerr << "Unable to create network. " << std::endl
              << e.what() << std::endl;
    return 1;
  } catch (const std::runtime_error &e) {
    std::cerr << "Unable to init. network. " << std::endl
              << e.what() << std::endl;
    return 1;
  }

  vector<double> colorVarVec;
  // loop through all images in the dataset
  for(size_t i=0; i<nrImages; ++i) {

    // find input image and labels
    string rgbImageName, labelImageName;
    dataset.getImage(i, rgbImageName);

    // predict image
    cv::Mat rgbImage = cv::imread(rgbImageName, cv::IMREAD_COLOR);
    status = net->infer(rgbImage, mask_argmax, prob_max, verbose);
    if (status != bonnet::CNN_OK) {
      std::cerr << "Failed to run CNN." << std::endl;
      return 1;
    }

    // convert xentropy mask to colors using dictionary
    cv::Mat mask_bgr(mask_argmax.rows, mask_argmax.cols, CV_8UC3);
    status = net->color(mask_argmax, mask_bgr, verbose);
    if (status != bonnet::CNN_OK) {
      std::cerr << "Failed to color result of CNN." << std::endl;
      return 1;
    }

    /*
     * Superpixel with bonnet
     */
    cv::Mat bonnetResultOverlay;
    UtilsFuses::alphaBlend(mask_bgr, rgbImage, 0.8, bonnetResultOverlay);
    cv::Mat inputCopy, converted, mask, labels;
    bonnetResultOverlay.copyTo(inputCopy);
    cv::cvtColor(inputCopy, converted, cv::COLOR_BGR2Lab);
    int region = sqrt((converted.cols)*(converted.rows)/spParam.nr_sp);
    cv::Ptr<cv::ximgproc::SuperpixelLSC> lsc = cv::ximgproc::
        createSuperpixelLSC(converted, region, spParam.ratio);
    lsc->iterate(spParam.sp_iterations);
    lsc->enforceLabelConnectivity(spParam.min_element_size);

    int nrNodes = lsc->getNumberOfSuperpixels();
    lsc->getLabels(labels);

    /*
     * front end
     */
    SegmentationFrontEnd sfe(rgbImage, segParam, labels, nrNodes, i, verbose);
    colorVarVec.push_back(sfe.colorVariance);

  }

  // write colorVarVec to csv
  ofstream varianceFile;
  varianceFile.open(outputFile);
  for(auto i=colorVarVec.begin(); i!=colorVarVec.end(); ++i) {
    varianceFile << *i << "\n";
  }
  varianceFile.close();

  // compute beta
  double averageVar;
  for (auto& n : colorVarVec) {
    averageVar += n;
  }
  averageVar = averageVar / colorVarVec.size();
  double beta = 1.0 / (2.0 * averageVar);
  cout << "Average color variance = " << averageVar << ", beta fitted = " << beta << endl;

  cout << "Fitting code completed successfully!" << endl;
}
#endif
