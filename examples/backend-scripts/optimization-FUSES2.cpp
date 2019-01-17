/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   optimization-FUSES2.cpp
 * @brief  compute optimization using Fuses2 (and load/save CRF to HDF5)
 * @author Siyi Hu, Luca Carlone
 */

#include "FUSES/Fuses2.h"
#include "FUSES/OpenGMParser.h"
#include "FUSES/Timer.h"
#include "FUSES/UtilsFuses.h"

using namespace std;
using namespace CRFsegmentation;
using namespace FUSES;
using namespace opengm;
using namespace myUtils::Timer;

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " [path/to/hdf5/file] "
        "[<optional>path/to/ground-truth/csv/file]" << endl;
    exit(1);
  }

  bool verbose = false;
  bool log_iterates = true;
  Rounding round = WinnerTakeAll;

  // compute optimization results
  OpenGMParser fg(argv[1]);
  CRF crf = fg.getCRF();

  // solve using Fuses2
  FUSESOpts options;
  options.rmax = crf.nrClasses + 10; // maximum rank in the staircase
  options.verbose = verbose;
  options.log_iterates = log_iterates;
  options.round = round;
  options.initializationType = Random; // IMPORTANT!!
  std::cout << "-- initializationType = Random" << std::endl;
  Fuses2 fs(crf, options);
  fs.solve();
  const FUSESResult& fsResult = fs.getResult();
  vector<size_t> labels = fs.convertMatrixToVec(fsResult.xhat); // from binary matrix to int vector

  //////////////////////// COMPUTE AND LOG RESULTS ///////////////////
  // log fval for rounded solutions at each iteration
  vector<double> fvalRounded = fs.computeIntermediateRoundedObjectives();

  // write results to csv
  ofstream outputFile;
  string outputName = argv[1];
  outputName.replace(outputName.end()-3, outputName.end(), "_FUSES2.csv"); // remove hdf5 extension and add _Fuses2
  outputFile.open(outputName);
  outputFile << "Total time(ms)," << fsResult.timing.total * 1000 /*ms*/ << "\n"
             << "Number of nodes,"  << crf.nrNodes << "\n"
             << "Number of classes,"  << crf.nrClasses << "\n";
  if (argc < 3) {
    outputFile << "Number of correct labels,"  << "N/A" << "\n";
  } else {
    outputFile << "Number of correct labels,"
        << crf.nrNodes - UtilsFuses::GetLabelDiff(labels, argv[2]) /*nr incorrect labels*/ << "\n";
  }
  outputFile << "Value after rounding,"  << fsResult.Fxhat << "\n";

  UtilsFuses::AppendLabelsToCSV(outputFile, labels);
  vector<double> iterations;
  const vector<double>& values = fsResult.function_values.back(); // vector of vector (we take last step of staircase)
  const vector<double>& times = fsResult.timing.elapsed_optimization.back(); // vector of vector (we take last step of staircase)
  UtilsFuses::AppendIterValTimeToCSV(outputFile, iterations, values, times,fvalRounded);
  outputFile.close();

  // print out final results
  cout << "Fuses2 optimal value = " << fsResult.Fxhat
      << " Fuses2 optimal value (MRF) = " << crf.evaluateMRF(labels)
      << ", time elapsed = " << fsResult.timing.total * 1000
      << " ms." << endl;
  fsResult.timing.print();
  cout << "Results saved to: " << outputName << endl;
  cout << "Fuses2 code completed successfully!" << endl;
}
