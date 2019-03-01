/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   fuses-bonnet-Cityscapes.cpp
 * @brief  compute optimization using fuses (with bonnet in the front end) on
 *         Cityscapes dataset
 * @author Siyi Hu, Luca Carlone
 */

#ifdef USE_BONNET

// fuses stuff
#include "FUSES/OpenGMParser.h"
#include "FUSES/CityscapesParser.h"
#include "FUSES/Fuses.h"
#include "FUSES/Fuses2DA.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/UtilsFuses.h"
#include "FUSES/UtilsGM.h"
#include "FUSES/Timer.h"

// c++ stuff
#include <chrono>
#include <iomanip>  // for setfill
#include <iostream>
#include <string>

// net stuff
#include <bonnet.hpp>
#include <IDToLabelConst.h>

using namespace std;
using namespace CRFsegmentation;
using namespace FUSES;
using namespace opengm;
using namespace myUtils::Timer;

int main(int argc, char **argv) {

  bool verbose = false;
  bool constUnary = false;
  bool log_iterates = false;
  float alpha = 0.6; // for alpha blending
  string resultFolder = "results";

  if (argc < 5) {
    cout << "Usage: " << argv[0] << " [path/to/dataset/directory] [nameOfDataset] [path/to/frozen/model] [path/to/param/file]" << endl;
    exit(1);
  }

  // setup output folder
  boost::filesystem::path resultPath(resultFolder);
  if (boost::filesystem::is_directory(resultPath)) {
    cout << "Directory: " << resultFolder << " already exists." << endl;
    exit(1);
  }
  else {
    boost::filesystem::create_directories(resultFolder);
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

  // loop through all images in the dataset
  for(size_t i=0; i<nrImages; ++i) {

    // find input image and labels
    string rgbImageName, labelImageName;
    dataset.getImage(i, rgbImageName);
    dataset.getLabel(i, labelImageName);
    cv::Mat rgbImage = cv::imread(rgbImageName, cv::IMREAD_COLOR);
    cv::Mat labelImage = cv::imread(labelImageName, cv::IMREAD_GRAYSCALE);

    // get output file names
    boost::filesystem::path p(rgbImageName);
    string frameName = p.filename().string();
    frameName = resultFolder + "/" + frameName;
    // eg.'<resultFolder>/aachen_000000_000019_leftImg8bit.png'

    string modelName = frameName;     // eg.'aachen_000000_000019.h5';
    modelName.replace(modelName.end()-16, modelName.end(), ".h5");

    string bonnetResult = frameName;     // eg.'aachen_000000_000019_bonnetResult.png';
    bonnetResult.replace(bonnetResult.end()-15, bonnetResult.end(), "bonnetResult.png");

    string fusesResult = frameName;     // eg.'aachen_000000_000019_fusesResult.png';
    fusesResult.replace(fusesResult.end()-15, fusesResult.end(), "fusesResult.png");

    string fuses2DAResult = frameName;     // eg.'aachen_000000_000019_fuses2DAResult.png';
    fuses2DAResult.replace(fuses2DAResult.end()-15, fuses2DAResult.end(), "fuses2DAResult.png");

    /*
     *  Front end
     */
    SegmentationFrontEnd sfe(rgbImage, segParam, spParam, spAlgorithm, i, verbose);

    // predict image
    cout << std::setfill('=') << std::setw(80) << "" << endl;
    cout << "Predicting image: " << rgbImageName << endl;
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
    sfe.getUnaryFactorsFromBonnet(mask_argmax, rgbImage.cols, rgbImage.rows, constUnary, verbose);
    const CRF& crf = sfe.getCRF();

    // save bonnet result image
    cv::Mat bonnetResultOverlay;
    UtilsFuses::alphaBlend(mask_bgr, rgbImage, alpha, bonnetResultOverlay);
    cv::imwrite(bonnetResult, bonnetResultOverlay);

    // save CRF to hdf5 file
    OpenGMParser fg(crf);
    fg.saveModel(modelName);

    /*
     * Back end
     */
    // load model
    OpenGMParser fg_be(modelName);
    const CRF crf_be = fg_be.getCRF();

    // FUSES
    FUSESOpts options1;
    options1.rmax = crf_be.nrClasses + 10; // maximum rank in the staircase
    options1.verbose = verbose;
    options1.log_iterates = log_iterates;
    options1.use_best_rounded = true;
    Fuses fs1(crf_be, options1);
    fs1.solve();
    const FUSESResult& fsResult1 = fs1.getResult();
    vector<size_t> labels1 = UtilsGM::GetLabelsFromMatrix(fsResult1.xhat, crf_be.nrNodes, crf_be.nrClasses);

    // print out final results
    cout << "Fuses optimal value = " << fsResult1.Fxhat
        << " Fuses optimal value (MRF) = " << crf_be.evaluateMRF(labels1)
        << ", time elapsed = " << fsResult1.timing.total * 1000
        << " ms." << endl;

    cv::Mat fsResultImage1, fsResultOverlay1;
    UtilsFuses::GetLabelImage(labels1, rgbImage, sfe.getSP(), dataset.getColorVec(), fsResultImage1);
    UtilsFuses::alphaBlend(fsResultImage1, rgbImage, alpha, fsResultOverlay1);
    cv::imwrite(fusesResult, fsResultOverlay1);

    // FUSES2DA
    FUSESOpts options2;
    options2.rmax = crf_be.nrClasses + 10; // maximum rank in the staircase
    options2.verbose = verbose;
    options2.precon = None;
    options2.log_iterates = log_iterates;
    options2.grad_norm_tol = 1e-2; // IMPORTANT FOR FUSES2DA
    options2.preconditioned_grad_norm_tol = 1e-2;
    options2.initializationType = Random; // IMPORTANT!!
    Fuses2DA fs2(crf_be, options2);
    fs2.alpha_init = 0.005;
    fs2.use_constant_stepsize = true;
    fs2.solveDA();
    const FUSESResult& fsResult2 = fs2.getResult();
    vector<size_t> labels2 = UtilsGM::GetLabelsFromMatrix(fsResult2.xhat, crf_be.nrNodes, crf_be.nrClasses);

    // print out final results
    cout << "Fuses2DA optimal value = " << fsResult2.Fxhat
        << " Fuses2DA optimal value (MRF) = " << crf_be.evaluateMRF(labels2)
        << ", time elapsed = " << fsResult2.timing.total * 1000
        << " ms." << endl;
    cv::Mat fsResultImage2, fsResultOverlay2;
    UtilsFuses::GetLabelImage(labels2, rgbImage, sfe.getSP(), dataset.getColorVec(), fsResultImage2);
    UtilsFuses::alphaBlend(fsResultImage2, rgbImage, alpha, fsResultOverlay2);
    cv::imwrite(fuses2DAResult, fsResultOverlay2);

  }

  cout << "Fuses code completed successfully!" << endl;
}
#endif
