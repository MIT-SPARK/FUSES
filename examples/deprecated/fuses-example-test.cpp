/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   fuses-hdf5.cpp
 * @brief  compute optimization using fuses (and load/save CRF to HDF5)
 * @author Siyi Hu, Luca Carlone
 */

#include "UtilsFuses.h"
#include "OpenGMParser.h"
#include "CityscapesParser.h"
#include "Fuses.h"
#include "SegmentationFrontEnd.h"
#include "Timer.h"

using namespace std;
using namespace CRFsegmentation;
using namespace FUSES;
using namespace opengm;
using namespace myUtils::Timer;

static const string DATASET_PATH = "/home/siyi/Documents/Research_code/sdpSegmentation/tests/data";
static const string testImgOriginal = DATASET_PATH + "/trainImage.jpg";
static const string testImgLabel = DATASET_PATH + "/trainLabel.png";
static const string paramFile = DATASET_PATH + "/fusesParameters.yaml";
static const string param_defaultFile = DATASET_PATH + "/fusesParameters_default.yaml";

int main(int argc, char **argv) {

  bool verbose = false;
  bool log_iterates = true;
  string textFileName = "tempNames.txt";

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " [path/to/dataset/directory] [nameOfDataset]" << endl;
    exit(1);
  }

  // cityscapes dataset parser
  CityscapesParser dataset(argv[1], argv[2]);
  size_t nrImages = dataset.rgbFiles_.size();

  // front end parameters
  int spAlgorithm = LSC;
  int pctClassified = 50;
  int maxSamplingIter = 100;
  SPParam spParam;
  SegParam segParam;
  loadParameters(spParam, segParam, paramFile);

  // initialize output text file (with model and ground-truth-labels file names)
  ofstream textFile;
  textFile.open(textFileName);

  // loop through all images in the dataset
  for(size_t i=0; i<1; ++i) {

    // find input image and labels
    string rgbImage, labelImage;
    dataset.getImage(i, rgbImage);
    dataset.getLabel(i, labelImage);

    // get output file names
    string targetString = argv[2];
    targetString += "_";
    size_t pos = rgbImage.find(targetString);
    string frameName = rgbImage.substr(pos); // eg.'aachen_000000_000019_leftImg8bit.png'

    string modelName = frameName;     // eg.'aachen_000000_000019.h5';
    modelName.replace(modelName.end()-16, modelName.end(), ".h5");

    string labelFileName = frameName; // eg.'aachen_000000_000019_labels.csv';
    labelFileName.replace(labelFileName.end()-15, labelFileName.end(), "labels.csv");

    // This saves the name file to different txt files for each frame
//    string textFileName = frameName; // eg.'aachen_000000_000019_names.txt';
//    textFileName.replace(textFileName.end()-15, labelFileName.end(), "names.txt");

    // front end
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

    // solve using fuses
    FUSESOpts options;
    options.rmax = crf.nrClasses + 10;
    options.verbose = verbose;
    options.log_iterates = log_iterates;
    Fuses fs(crf, options);
    fs.solve();
    FUSESResult& fsResult = fs.result_;
    vector<size_t> labels = fs.convertMatrixToVec(fsResult.xhat);

    vector<double> fvalRounded;
    if(log_iterates) { // log function value at each iteration after rounding
      for(auto j=fsResult.iterates.back().begin(); j!=fsResult.iterates.back().end(); ++j) {
        // Evaluate objective function at ROUNDED solution
        gtsam::Matrix Xhat(crf.nrNodes + crf.nrClasses, crf.nrClasses);
        Xhat.block(0, 0, crf.nrNodes, crf.nrClasses) = fs.round_solution(*j);
        Xhat.block(crf.nrNodes, 0, crf.nrClasses, crf.nrClasses) =
            gtsam::Matrix::Identity(crf.nrClasses, crf.nrClasses);

        fvalRounded.push_back(fs.evaluate_objective(Xhat));
      }

      if(fvalRounded.size() != fsResult.function_values.back().size()) {
        fvalRounded.push_back(fsResult.Fxhat);
      }
    }

    // write to csv
    ofstream outputFile;
    string outputName = modelName;
    outputName.replace(outputName.end()-3, outputName.end(), "_FUSES.csv");
    outputFile.open(outputName);
    outputFile << "Total time(ms)," << fsResult.timing.total * 1000
          << "\n"
        << "Number of nodes,"  << crf.nrNodes << "\n"
        << "Number of classes,"  << crf.nrClasses << "\n"
        << "Number of correct labels,"
        << crf.nrNodes - UtilsFuses::GetLabelDiff(labels, argv[2]) << "\n"
        << "Value after rounding,"  << fsResult.Fxhat << "\n";

    UtilsFuses::AppendLabelsToCSV(outputFile, labels);
    vector<double> iterations;
    vector<double>& values = fsResult.function_values.back();
    vector<double>& times = fsResult.timing.elapsed_optimization.back();
    if(log_iterates) {
      UtilsFuses::AppendIterValTimeToCSV(outputFile, iterations, values, times,
          fvalRounded);
    } else {
      UtilsFuses::AppendIterValTimeToCSV(outputFile, iterations, values, times);
    }
    outputFile.close();

    // print out final results
    cout << "Fuses optimal value = " << fsResult.Fxhat
            << ", time elapsed = " << fsResult.timing.total * 1000
            << " ms." << endl;

    cout << "Results saved to: " << outputName << endl;

    // write model name and ground-truth-labels file name to a text file
    textFile << modelName << "\n" << labelFileName << "\n" ;

  }

  // close text file
  textFile.close();

  cout << "Fuses code completed successfully!" << endl;
}
