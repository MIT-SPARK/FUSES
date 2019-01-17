/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   fuses-fittingBeta.cpp
 * @brief  log variance of color difference for all images in a specified
 *         Cityscapes dataset folder
 * @author Siyi Hu, Luca Carlone
 */

#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <numeric>
#include <math.h>
#include "FUSES/SegmentationFrontEnd.h"
#include "FUSES/CityscapesParser.h"
#include "FUSES/Timer.h"

using namespace std;
using namespace CRFsegmentation;
using namespace myUtils;

int main(int argc, char **argv) {

  int verbose = 0;
  string outputFile = "colorVariances.csv";

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

  vector<double> colorVarVec;
  for(size_t i=0; i<nrImages; ++i){

    // Front End
    string rgbImageName;
    dataset.getImage(i, rgbImageName);

    cv::Mat rgbImage = cv::imread(rgbImageName);
    SegmentationFrontEnd sfe(rgbImage, segParam, spParam, spAlgorithm, i, verbose);
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
