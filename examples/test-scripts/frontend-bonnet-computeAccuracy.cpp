/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   frontend-bonnet-computeAccuracy.cpp
 * @brief  compute accuracy statistics for bonnet as a base line for FUSES
 * @author Siyi Hu, Luca Carlone
 */

#include "FUSES/UtilsFuses.h"
#include "FUSES/Timer.h"
#include <boost/filesystem.hpp>

using namespace std;
using namespace myUtils::Timer;

int main(int argc, char **argv) {

  bool verbose = false;
  const string toLabelFolder = "/gtFine_trainvaltest/gtFine/val/";

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " [path/to/dataset/directory] [path/to/bonnet/image]" << endl;
    exit(1);
  }

  // get ground-truth label path
  string bonnetFileName = argv[2];
  boost::filesystem::path p(bonnetFileName);
  string frameName = p.filename().string(); // eg.'aachen_000000_000019_bonnet.png'
  string targetString = "_";
  size_t pos = frameName.find(targetString);
  string datasetName = frameName.substr(0, pos); // eg. 'aachen'
  string labelFileName = argv[1] + toLabelFolder + datasetName + "/" + frameName;
          // eg.'path/to/label/directory/aachen_000000_000019_gtFine_labelIds.png'
  labelFileName.replace(labelFileName.end()-10, labelFileName.end(), "gtFine_labelIds.png");

  // output file name: save _bonnetAccuracy.csv in the same location as _bonnet.png
  string accuracyFileName = bonnetFileName;
  accuracyFileName.replace(accuracyFileName.end()-10, accuracyFileName.end(), "bonnetAccuracy.csv");

  // compute accuracy
  auto start_time = myUtils::Timer::tic();
  UtilsFuses::ComputeBonnetAccuracy(labelFileName, bonnetFileName, accuracyFileName);
  cout << "time elapsed computing accuracy matrix: " << myUtils::Timer::toc(start_time) << endl;

  cout << "Front end code completed successfully!" << endl;
}
