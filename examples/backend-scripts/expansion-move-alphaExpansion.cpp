/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   expansion-move-alphaExpansion.cpp
 * @brief  compute optimization using alpha-expansion algorithm
 * @author Siyi Hu, Luca Carlone
 */

#include "FUSES/UtilsFuses.h"
#include "FUSES/OpenGMParser.h"
#include "FUSES/Timer.h"

using namespace std;
using namespace CRFsegmentation;
using namespace opengm;
using namespace myUtils::Timer;

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " [path/to/hdf5/file]"
        " [<optional>path/to/ground-truth/csv/file]" << endl;
    exit(1);
  }

  bool verbose = false;

  // compute optimization results
  OpenGMParser fg(argv[1]);

  vector<size_t> labels;
  std::vector<double> iterations, values, times;
  auto start_time = tic();
  fg.computeAE(labels, iterations, values, times, verbose);
  double end_time = toc(start_time);

  // write to csv
  ofstream outputFile;
  string outputName = argv[1];
  outputName.replace(outputName.end()-3, outputName.end(), "_AE.csv");
  outputFile.open(outputName);
  outputFile << "Total time(ms),"  << end_time * 1000 << "\n";
  outputFile << "Number of nodes,"  << fg.nrNodes() << "\n";
  outputFile << "Number of classes,"  << fg.nrClasses() << "\n";
  if (argc < 3) {
    outputFile << "Number of correct labels,"  << "N/A" << "\n";
  } else {
    outputFile << "Number of correct labels,"
        << fg.nrNodes() - UtilsFuses::GetLabelDiff(labels, argv[2]) << "\n";
  }
  UtilsFuses::AppendLabelsToCSV(outputFile, labels);
  UtilsFuses::AppendIterValTimeToCSV(outputFile, iterations, values, times);
  outputFile.close();

  // print out final results
  cout << "Alpha-expansion optimal value = " << fg.evaluate(labels)
      << ", time elapsed = " << end_time*1000 << " ms." << endl;

  cout << "Results saved to: " << outputName << endl;

  cout << "Alpha-expansion code completed successfully!" << endl;
}
