/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   frontend-rewrite-hdf5.cpp
 * @brief  take crf stored in the input hdf5 files and change the weights of the
 *         binary factor as (a + b*initialWeight)
 * @author Siyi Hu, Luca Carlone
 */

#include "FUSES/OpenGMParser.h"
#include "FUSES/CRF.h"

using namespace std;
using namespace CRFsegmentation;
using namespace opengm;

int main(int argc, char **argv) {

  if (argc < 5) {
    cout << "Usage: " << argv[0] << " [path/to/input/hdf5/file] [path/to/output/hdf5/file] [parameter_a] [parameter_b]" << endl;
    exit(1);
  }

  double a = stod(argv[3]);
  double b = stod(argv[4]);

  // compute optimization results
  OpenGMParser fg_in(argv[1]);
  CRF crf = fg_in.getCRF();

  // adjust weights of the binary factors
  for (auto& bf : crf.binaryFactors) {
    bf.weight = a + b * bf.weight;
  }

  // save CRF to hdf5 file
  OpenGMParser fg_out(crf);
  fg_out.saveModel(argv[2]);
}
