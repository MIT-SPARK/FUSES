/**
 * @file   fuses-bonnet-batch.cpp
 * @brief  compute bonnet labels and optionally optimized using fuses for all
 *         images in given directory
 * @author Siyi Hu
 */

// c++ stuff
#include <stdlib.h>
#include <iostream>
#include <iomanip>  // for setfill
#include <string>
#include <experimental/optional>

// fuses stuff
#include "FUSES/Fuses.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/UtilsFuses.h"
#include "FUSES/UtilsGM.h"
#include "FUSES/Timer.h"

// net stuff
#include <bonnet.hpp>
#include <IDToLabelConst.h>

// other libraries
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>

using namespace std;
using namespace CRFsegmentation;
using namespace FUSES;

static const string bonnetResultFolder = "results/bonnet/segmented";
static const string bonnetOverlayFolder = "results/bonnet/overlays";
static const string fusesResultFolder = "results/fuses/segmented";
static const string fusesOverlayFolder = "results/fuses/overlays";

struct path_leaf_string {
  string operator()(const boost::filesystem::directory_entry& entry) const{
    return entry.path().leaf().string();
  }
};

int main(int argc, char **argv) {
  bool verbose = false;
  float alpha = 0.6; // for alpha blending

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " [path/to/image/directory] "
        "[path/to/frozen/model] [<optional>path/to/param/file]" << endl;
    exit(1);
  } else {
    cout << "Reading images from folder: " << argv[1] << endl;
    cout << "Bonnet model path: " << argv[2] << endl;
  }

  // setup output folder
  boost::filesystem::path bonnetResultPath(bonnetResultFolder);
  if (boost::filesystem::is_directory(bonnetResultPath)) {
    cout << "Directory: " << bonnetResultFolder << " already exists." << endl;
  }
  else {
    boost::filesystem::create_directories(bonnetResultPath);
    cout << "Create directory: " << bonnetResultFolder << " ." << endl;
  }

  boost::filesystem::path bonnetOverlayPath(bonnetOverlayFolder);
  if (boost::filesystem::is_directory(bonnetOverlayPath)) {
    cout << "Directory: " << bonnetOverlayFolder << " already exists." << endl;
  }
  else {
    boost::filesystem::create_directories(bonnetOverlayPath);
    cout << "Create directory: " << bonnetOverlayFolder << " ." << endl;
  }

  if (argc > 3) { // running fuses as well
    boost::filesystem::path fusesResultPath(fusesResultFolder);
    if (boost::filesystem::is_directory(fusesResultPath)) {
      cout << "Directory: " << fusesResultFolder << " already exists." << endl;
    }
    else {
      boost::filesystem::create_directories(fusesResultPath);
      cout << "Create directory: " << fusesResultFolder << " ." << endl;
    }

    boost::filesystem::path fusesOverlayPath(fusesOverlayFolder);
    if (boost::filesystem::is_directory(fusesOverlayPath)) {
      cout << "Directory: " << fusesOverlayFolder << " already exists." << endl;
    }
    else {
      boost::filesystem::create_directories(fusesOverlayPath);
      cout << "Create directory: " << fusesOverlayFolder << " ." << endl;
    }
  }

  // get paths to images
  vector<string> images;
  boost::filesystem::path inputPath(argv[1]);
  boost::filesystem::directory_iterator start(inputPath);
  boost::filesystem::directory_iterator end;
  transform(start, end, back_inserter(images), path_leaf_string());
  size_t nrImages = images.size();

  // network stuff
  cv::Mat mask_argmax, prob_max;
  bonnet::retCode status;
  std::unique_ptr<bonnet::Bonnet> net;
  string device = "/gpu:0";
  string backend = "trt";
  string path = argv[2];

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

  // get fuses frontend parameters and color vector(only used when argc > 2)
  int spAlgorithm = LSC;
  SPParam spParam;
  SegParam segParam;
  vector<cv::Vec3b> colorVec;
  if (argc > 3) {
    string paramFile = argv[3];
    loadParameters(spParam, segParam, paramFile);
    std::string dataFile = path + "/data.yaml";
    UtilsFuses::InitColor(dataFile, colorVec);
  }

  // loop through all images in the dataset
  for(size_t i=0; i<nrImages; ++i) {

    string rgbImageName = string(argv[1]) + '/' + images[i];
    cv::Mat rgbImage = cv::imread(rgbImageName, cv::IMREAD_COLOR);

    // get output file names
    boost::filesystem::path p(rgbImageName);
    string frameName = p.filename().string();
    string bonnetResultName = bonnetResultFolder + "/" + frameName;
    string bonnetOverlayName = bonnetOverlayFolder + "/" + frameName;

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
    cv::imwrite(bonnetResultName, mask_bgr);

    // overlay bonnet labels with original image
    cv::Mat bonnetResultOverlay;
    UtilsFuses::alphaBlend(mask_bgr, rgbImage, alpha, bonnetResultOverlay);
    cv::imwrite(bonnetOverlayName, bonnetResultOverlay);

    // optimize using fuses
    if (argc > 3) {
      // front end
      SegmentationFrontEnd sfe(rgbImage, segParam, spParam, spAlgorithm, i, verbose);
      sfe.getUnaryFactorsFromBonnet(mask_argmax, prob_max, rgbImage.cols,
          rgbImage.rows, verbose);
      const CRF& crf = sfe.getCRF();

      // FUSES
      FUSESOpts options;
      options.rmax = crf.nrClasses + 10; // maximum rank in the staircase
      options.verbose = verbose;
      options.use_best_rounded = true;
      Fuses fs(crf, options);
      fs.solve();
      const FUSESResult& fsResult = fs.getResult();
      vector<size_t> labels = UtilsGM::GetLabelsFromMatrix(fsResult.xhat, crf.nrNodes, crf.nrClasses);

      // print out final results
      cout << "Fuses optimal value = " << fsResult.Fxhat
          << ", time elapsed = " << fsResult.timing.total * 1000
          << " ms." << endl;

      string fusesResultName = fusesResultFolder + "/" + frameName;
      string fusesOverlayName = fusesOverlayFolder + "/" + frameName;

      cv::Mat fusesResultImage, fusesResultOverlay;
      UtilsFuses::GetLabelImage(labels, rgbImage, sfe.getSP(), colorVec, fusesResultImage);
      cv::imwrite(fusesResultName, fusesResultImage);

      UtilsFuses::alphaBlend(fusesResultImage, rgbImage, alpha, fusesResultOverlay);
      cv::imwrite(fusesOverlayName, fusesResultOverlay);
    }

  }

  cout << "Finished looping through all images!" << endl;
}
