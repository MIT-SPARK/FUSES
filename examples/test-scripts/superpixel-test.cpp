/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   superpixel-test.cpp
 * @brief  test quality of superpixel algorithm
 * @author Siyi Hu, Luca Carlone
 */

#include "FUSES/CityscapesParser.h"
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/Timer.h"
#include "FUSES/UtilsFuses.h"
#include "IDToLabelConst.h"

using namespace std;
using namespace CRFsegmentation;
using namespace myUtils::Timer;

int main(int argc, char **argv) {

  bool verbose = false;
  bool log_iterates = true;
  string textFileName = "tempNames-SP.txt";
  string logFileName = "initializationTime.csv";

  if (argc < 4) {
    cout << "Usage: " << argv[0] << " [path/to/dataset/directory] [nameOfDataset] [path/to/param/file]" << endl;
    exit(1);
  }

  // cityscapes dataset parser
  CityscapesParser dataset(argv[1], argv[2]);
  size_t nrImages = dataset.nrImages();

  // front end parameters
  int spAlgorithm = LSC;
  SPParam spParam;
  SegParam segParam;
  string paramFile = argv[3];
  loadParameters(spParam, segParam, paramFile);

  // initialize output text file (with ground-truth-labels file names)
  ofstream textFile;
  textFile.open(textFileName);

  ofstream logFile;
  logFile.open(logFileName);

  // loop through all images in the dataset
  for(size_t i=0; i<nrImages; ++i) {

    // find input image and labels
    string rgbImageName, labelImageName;
    dataset.getImage(i, rgbImageName);
    dataset.getLabel(i, labelImageName);

    // get output file names
    string targetString = argv[2];
    targetString += "_";
    size_t pos = rgbImageName.find(targetString);
    string frameName = rgbImageName.substr(pos); // eg.'aachen_000000_000019_leftImg8bit.png'

    string labelFileName = frameName; // eg.'aachen_000000_000019_labels.csv';
    labelFileName.replace(labelFileName.end()-15, labelFileName.end(), "labels.csv");

    string pixelLabelFileName = frameName; // eg.'aachen_000000_000019_pixelLabels.csv'; (pixel-level info)
    pixelLabelFileName.replace(pixelLabelFileName.end()-15, pixelLabelFileName.end(), "pixelLabels.csv");

    /*
     * front end
     */
    cv::Mat rgbImage = cv::imread(rgbImageName, cv::IMREAD_COLOR);
    cv::Mat labelImage = cv::imread(labelImageName, cv::IMREAD_COLOR);
    auto sfe_start = myUtils::Timer::tic();
    SegmentationFrontEnd sfe(rgbImage, segParam, spParam, spAlgorithm, i, verbose);
    double sfe_end = myUtils::Timer::toc(sfe_start);
    sfe.getGTLabelsFromID(labelImage, 20, labelVec, verbose);

    // write ground-truth label to csv
    vector<size_t> labelsGT = sfe.getSPLabels();
    ofstream labelFile;
    labelFile.open(labelFileName);
    for(auto i=labelsGT.begin(); i!=labelsGT.end(); ++i) {
      labelFile << *i << "\n";
    }
    labelFile.close();

    // save ground-truth labels
    UtilsFuses::LogSuperpixelGTLabels(pixelLabelFileName, labelImageName, 20, labelVec, sfe.getSP());

    // write ground-truth labels file name and pixel-level label file name to a text file
    textFile << labelFileName << "\n" << pixelLabelFileName << "\n" ;

    // write sfe initialization time to a csv file
    logFile << sfe_end << "\n" ;

  }

  // close text file
  textFile.close();

  cout << "Front end code completed successfully!" << endl;
}
