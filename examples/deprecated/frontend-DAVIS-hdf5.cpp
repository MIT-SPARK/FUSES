/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   frontend-DAVIS-hdf5.cpp
 * @brief  compute and save crf as well as correct labels for later use
 * @author Siyi Hu, Luca Carlone
 */

#include "OpenGMParser.h"
#include "DavisParser.h"
#include "SegmentationFrontEnd.h"
#include "Timer.h"

using namespace std;
using namespace CRFsegmentation;
using namespace opengm;
using namespace myUtils::Timer;

int main(int argc, char **argv) {

  bool verbose = false;
  bool log_iterates = true;
  string textFileName = "tempNames.txt";

  // Unary term parameters
  int pctClassified = 50;
  int maxSamplingIter = 100;

  if (argc < 4) {
    cout << "Usage: " << argv[0] << " [path/to/dataset/directory] [nameOfDataset] [path/to/param/file]" << endl;
    exit(1);
  }

  // cityscapes dataset parser
  DavisParser dataset(argv[1], argv[2]);
  size_t nrImages = dataset.rgbFiles_.size();

  // front end parameters
  int spAlgorithm = LSC;
  SPParam spParam;
  SegParam segParam;
  string paramFile = argv[3];
  loadParameters(spParam, segParam, paramFile);

  // initialize output text file (with model and ground-truth-labels file names)
  ofstream textFile;
  textFile.open(textFileName);

  // loop through all images in the dataset
  for(size_t i=0; i<10; ++i) {

    // find input image and labels
    string rgbImage, labelImage;
    dataset.getImage(i, rgbImage);
    dataset.getLabel(i, labelImage);

    // get output file names
    size_t pos = rgbImage.size() - 9;
    string frameName = rgbImage.substr(pos); // eg.'00000.jpg'
    cout << frameName << endl;

    string modelName = frameName;     // eg.'00000.h5';
    modelName.replace(modelName.end()-4, modelName.end(), ".h5");

    string labelFileName = frameName; // eg.'aachen_000000_000019_labels.csv';
    labelFileName.replace(labelFileName.end()-4, labelFileName.end(), "_labels.csv");

    /*
     * front end
     */
    SegmentationFrontEnd sfe(rgbImage, segParam, spParam, spAlgorithm, i, verbose);
    sfe.getUnaryFactorsFromImage(labelImage, pctClassified, maxSamplingIter, verbose);

    CRF crf = sfe.getCRF();

    // write ground-truth label to csv
    vector<int> labelsGT = sfe.getSPLabels();
    ofstream labelFile;
    labelFile.open(labelFileName);
    for(auto i=labelsGT.begin(); i!=labelsGT.end(); ++i) {
      labelFile << *i << "\n";
    }
    labelFile.close();

    // save CRF to hdf5 file
    OpenGMParser fg(crf);
    fg.saveModel(modelName);

    // write model name and ground-truth-labels file name to a text file
    textFile << modelName << "\n" << labelFileName << "\n" ;

  }

  // close text file
  textFile.close();

  cout << "Front end code completed successfully!" << endl;
}
