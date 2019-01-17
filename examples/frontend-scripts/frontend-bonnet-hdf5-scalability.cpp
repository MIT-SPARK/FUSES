/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   frontend-bonnet-hdf5-fitting.cpp
 * @brief  take superpixel param num_sp and number of classes as input
 *         compute and save crf as well as correct labels for later use
 * @author Siyi Hu, Luca Carlone
 */

#ifdef USE_BONNET
#include "FUSES/OpenGMParser.h"
#include "FUSES/CityscapesParser.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/Timer.h"
#include "bonnet.hpp"
#include "IDToLabelConst.h"

using namespace std;
using namespace CRFsegmentation;
using namespace opengm;
using namespace myUtils::Timer;

int main(int argc, char **argv) {

  bool verbose = false;
  bool log_iterates = true;
  string textFileName = "tempNames-scalability.txt";
  string resultFolder = "results-scalability";

  if (argc < 6) {
    cout << "Usage: " << argv[0] << " [path/to/dataset/directory] "
        "[nameOfDataset] [path/to/frozen/model] [path/to/param/file]"
        " [num_sp] [<optional>num_classes]" << endl;
    exit(1);
  }

  if (argc > 6) {
    cout << "nrClasses = " << stoi(argv[6]) << endl;
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
  cout << "nr_sp = " << spParam.nr_sp << endl;

  // initialize output text file (with model and ground-truth-labels file names)
  ofstream textFile;
  textFile.open(textFileName);

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

    string bonnetImgName = frameName;     // eg.'<resultFolder>/aachen_000000_000019_bonnet.png';
    bonnetImgName.replace(bonnetImgName.end()-15, bonnetImgName.end(), "bonnet.png");

    string pixelLabelFileName = frameName; // eg.'<resultFolder>/aachen_000000_000019_pixelLabels.csv'; (pixel-level info)
    pixelLabelFileName.replace(pixelLabelFileName.end()-15, pixelLabelFileName.end(), "pixelLabels.csv");

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
    const CRF& crf = sfe.getCRF();

    // Reduce the number of classes is specified
    if(argc > 6) {
      size_t targetNrClasses = stoi(argv[6]);
      CRF crfReduced = crf;
      crfReduced.reduceNrClassesCRF(targetNrClasses);

      // write ground-truth label to csv
      const vector<size_t>& labelsGT = sfe.getSPLabels();
      ofstream labelFile;
      labelFile.open(labelFileName);
      for(auto i=labelsGT.begin(); i!=labelsGT.end(); ++i) {
        labelFile << *i - (*i/targetNrClasses)*targetNrClasses << "\n";
      }
      labelFile.close();

      // save CRF to hdf5 file
      OpenGMParser fg(crfReduced);
      fg.saveModel(modelName);
    } else {
      // write ground-truth label to csv
      const vector<size_t>& labelsGT = sfe.getSPLabels();
      ofstream labelFile;
      labelFile.open(labelFileName);
      for(auto i=labelsGT.begin(); i!=labelsGT.end(); ++i) {
        labelFile << *i << "\n";
      }
      labelFile.close();

      // save CRF to hdf5 file
      OpenGMParser fg(crf);
      fg.saveModel(modelName);
    }

    // write model name and ground-truth-labels file name to a text file
    textFile << modelName << "\n" << labelFileName << "\n" ;

  }

  // close text file
  textFile.close();

  cout << "Front end code completed successfully!" << endl;
}
#endif
