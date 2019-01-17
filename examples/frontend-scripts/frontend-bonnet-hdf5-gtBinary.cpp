/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   frontend-bonnet-hdf5-gtBinary.cpp
 * @brief  use ground truth for binary labels
 *         compute and save crf as well as correct labels for later use
 * @author Siyi Hu, Luca Carlone
 */

#ifdef USE_BONNET
#include "FUSES/OpenGMParser.h"
#include "FUSES/CityscapesParser.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/Timer.h"
#include "FUSES/UtilsFuses.h"
#include "bonnet.hpp"
#include "FUSES/IDToLabelConst.h"

using namespace std;
using namespace CRFsegmentation;
//using namespace FUSES;
using namespace opengm;
using namespace myUtils::Timer;

int main(int argc, char **argv) {

  bool verbose = false;
  bool useEdge = false;
  bool log_iterates = false;
  float alpha = 0.6; // for alpha blending
  string textFileName = "tempNames.txt";
  string resultFolder = "results-gtBinary";

  if (argc < 5) {
    cout << "Usage: " << argv[0] << " [path/to/dataset/directory] [nameOfDataset] [path/to/frozen/model] [path/to/param/file]" << endl;
    exit(1);
  }

  // setup output folder
  boost::filesystem::path resultPath(resultFolder);
  if (boost::filesystem::is_directory(resultPath)) {
    cout << "Directory: " << resultFolder << " already exists." << endl;
    cout << "Over-writing..." << endl;
//    exit(1);
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

  // initialize output text file (with model and ground-truth-labels file names)
  ofstream textFile;
  textFile.open(resultFolder + "/" + textFileName);

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

    string modelName = frameName;     // eg.'<resultFolder>/aachen_000000_000019.h5';
    modelName.replace(modelName.end()-16, modelName.end(), ".h5");

    string labelFileName = frameName; // eg.'<resultFolder>/aachen_000000_000019_labels.csv';
    labelFileName.replace(labelFileName.end()-15, labelFileName.end(), "labels.csv");

    string pixelLabelFileName = frameName; // eg.'<resultFolder>/aachen_000000_000019_pixelLabels.csv'; (pixel-level info)
    pixelLabelFileName.replace(pixelLabelFileName.end()-15, pixelLabelFileName.end(), "pixelLabels.csv");

    string bonnetImgName = frameName;     // eg.'<resultFolder>/aachen_000000_000019_bonnet.png';
    bonnetImgName.replace(bonnetImgName.end()-15, bonnetImgName.end(), "bonnet.png");

    // predict image
    status = net->infer(rgbImage, mask_argmax, prob_max, verbose);
    if (status != bonnet::CNN_OK) {
      std::cerr << "Failed to run CNN." << std::endl;
      return 1;
    }
    cv::imwrite(bonnetImgName, mask_argmax);

    /*
     * front end
     */
    SegmentationFrontEnd sfe(rgbImage, segParam, spParam, spAlgorithm, i, verbose);
    sfe.getGTLabelsFromID(labelImage, dataset.nrClasses(), labelVec, verbose);
    sfe.getUnaryFactorsFromBonnet(mask_argmax, prob_max, rgbImage.cols, rgbImage.rows, verbose);

    // correct binary
    const vector<size_t>& labelsGT = sfe.getSPLabels();
    sfe.correctBinary(labelsGT);

    /*
     * output files
     */
    // write ground-truth label to csv
    UtilsFuses::SaveLabels(labelFileName, labelsGT);

    // save CRF to hdf5 file
    const CRF& crf = sfe.getCRF();
    OpenGMParser fg(crf);
    fg.saveModel(modelName);

    // save ground-truth labels
    UtilsFuses::LogSuperpixelGTLabels(pixelLabelFileName, labelImageName,
        crf.nrClasses, labelVec, sfe.getSP());


    // write model name and ground-truth-labels file name to a text file
    textFile << modelName << "\n" << labelFileName << "\n" ;

    cout << "Front end files generated for frame " << i << ".\n";

  }

  // close text file
  textFile.close();

  cout << "Front end code completed successfully!" << endl;
}
#endif
