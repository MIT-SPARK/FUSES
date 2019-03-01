/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   FusesUtils.h
 * @brief  Utilities to analyze and save Fuses outputs
 * @author Siyi Hu, Luca Carlone
 */

#ifndef UTILSFUSES_H_
#define UTILSFUSES_H_

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <boost/optional.hpp>
#include <yaml-cpp/yaml.h>
#include "FUSES/SegmentationFrontEnd.h"

namespace UtilsFuses {
  /* ----------------------------------------------------------------------- */
  // Read labels from a csv file (assume data is stored in column)
  void GetLabelsFromCSV(const std::string& labelsFile,
      std::vector<size_t>* labels) {
    // load csv file
    std::ifstream file(labelsFile);
    if (!file.good()) {
      std::cout <<  "Could not read file " << labelsFile << std::endl;
      throw std::runtime_error("File not found.");
    }

    // read data
    std::string line;
    labels->clear();
    while (getline(file, line)) {
      labels->push_back(stoi(line));
    }
    file.close();
  }

  /* ---------------------------------------------------------------------- */
  // compute the number of label differences
  static int GetLabelDiff(const std::vector<size_t>& labels,
      const std::vector<size_t>& labelsGT) {
    if(labels.size() != labelsGT.size()) {
      std::cout <<  "Could not compare labelings ("<< labels.size()
                      << " labels computed and " << labelsGT.size()
                      << " labels from ground-truth)" << std::endl;
      throw std::runtime_error("SIZE mismatch");
    }
    int labelDiff = 0;
    for(size_t i=0; i<labels.size(); ++i) {
      if(labels[i]-labelsGT[i] != 0) {
        labelDiff++;
      }
    }
    return labelDiff;
  }

  /* ----------------------------------------------------------------------- */
  static int GetLabelDiff(const std::vector<size_t>& labels,
      const std::string& labelsGTFile) {
    std::vector<size_t> labelsGT;
    GetLabelsFromCSV(labelsGTFile, &labelsGT);
    return GetLabelDiff(labels, labelsGT);
  }

  /* ----------------------------------------------------------------------- */
  // Write solution to the given output file
  void AppendLabelsToCSV(std::ofstream& outputFile,
      const std::vector<size_t>& labels) {
    outputFile << "Solution";
    for(auto i=labels.begin(); i!= labels.end(); ++i) {
      outputFile << "," << *i;
    }
    outputFile << "\n";
  }

  /* ----------------------------------------------------------------------- */
  // Write iteration, value, time in columns to the given output file
  void AppendIterValTimeToCSV(std::ofstream& outputFile,
      const std::vector<double>& iterations, const std::vector<double>& values,
      const std::vector<double>& times,
      const boost::optional<std::vector<double>&> valuesExtra = boost::none) {
    outputFile << "Iteration,Value,Time(ms)\n";

    if(iterations.size()==0) {
      if (valuesExtra) { // valuesExtra provided
        for(size_t i = 0; i<values.size(); ++i) {
          outputFile << i << "," << values[i] << ","
            << times[i] * 1000 << "," << (*valuesExtra)[i] << "\n";
        }
      } else {           // both not provided
        for(size_t i = 0; i<values.size(); ++i) {
          outputFile << i << "," << values[i] << ","
            << times[i] * 1000 << "\n";
        }
      }
    } else {
      if (valuesExtra) { // iterations and valuesExtra provided
        for(size_t i = 0; i<values.size(); ++i) {
          outputFile << iterations[i] << "," << values[i] << ","
            << times[i] * 1000 << "," << (*valuesExtra)[i] << "\n";
        }
      } else {           // iterations provided
        for(size_t i = 0; i<values.size(); ++i) {
          outputFile << iterations[i] << "," << values[i] << ","
            << times[i] * 1000 << "\n";
        }
      }
    }
  }

  /* ----------------------------------------------------------------------- */
  std::pair<std::vector<double>,std::vector<double>> GetValuesAndCumulativeTimes(
      const std::vector<std::vector<double>>& values,
      const std::vector<std::vector<double>>&  elapsed_times){
    if(values.size() != elapsed_times.size()){
      throw std::runtime_error("FusesUtils: inconsistent size of values and times");
    }
    std::vector<double> valuesAs1Vector;
    for(size_t step=0; step<values.size(); step++){
      for(size_t i=0; i<values[step].size(); i++){
        valuesAs1Vector.push_back(values[step][i]);
      }
    }
    std::vector<double> timesAs1Vector;
    double cumTime = 0;
    for(size_t step=0; step<elapsed_times.size(); step++){
      for(size_t i=0; i<elapsed_times[step].size(); i++){
        timesAs1Vector.push_back(elapsed_times[step][i] + cumTime);
      }
      cumTime += elapsed_times[step].back();
    }
    return std::make_pair(valuesAs1Vector,timesAs1Vector);
  }

  /* ----------------------------------------------------------------------- */
  // Generate label image
  void GetLabelImage(const std::vector<size_t>& label,
      const cv::Mat& refImage,
      const std::vector<CRFsegmentation::SuperPixel>& superPixels,
      const std::vector<cv::Vec3b>& label_to_bgr,
      cv::Mat& resultImage) {
    cv::Mat temp(refImage.rows, refImage.cols, CV_8UC3, cv::Scalar(0,0,0));

    // loop through each superpixel
    for(int i=0; i<label.size(); ++i) {
      const CRFsegmentation::SuperPixel& sp = superPixels[i];
      size_t c = label[i];
      const cv::Vec3b& color = label_to_bgr[c];

      // set each pixel in the superpixel to the same color
      for(auto& pixel : sp.pixels) {
        temp.at<cv::Vec3b>(pixel.first, pixel.second) = color;
      }
    }
    temp.copyTo(resultImage);
  }

  /* ----------------------------------------------------------------------- */
  // Visualize image
  void Visualize(const std::vector<size_t>& label,
      const cv::Mat& refImage,
      const std::vector<CRFsegmentation::SuperPixel>& superPixels,
      const std::vector<cv::Vec3b>& label_to_bgr) {

    // check input dimensions
    if(label.size() != superPixels.size()) {
      std::cout <<  "Could not visualize image" << std::endl;
      throw std::runtime_error("The size of label vector does not match the "
          "number of superpixels.");
    }

    // result image
    cv::Mat resultImage;
    GetLabelImage(label, refImage, superPixels, label_to_bgr, resultImage);

    cv::imshow("segmented", resultImage);
    cv::waitKey(0);
  }

  /* ----------------------------------------------------------------------- */
  // Alpha blending of two images
  void alphaBlend(const cv::Mat& foreground, const cv::Mat& background,
      float alpha, cv::Mat& outImage) {
    // resize
    cv::Mat foregroundCopy;
    cv::resize(foreground, foregroundCopy, background.size(), 0, 0,
        cv::INTER_NEAREST);

    cv::addWeighted(foregroundCopy, alpha, background, 1.0-alpha, 0.0, outImage);
  }

  /* ----------------------------------------------------------------------- */
  // Setup labelToBGR vector given a yaml data file
  void InitColor(const std::string& dataFile,
      std::vector<cv::Vec3b>& labelToBGR) {
    YAML::Node cfg_data;
    try {
      cfg_data = YAML::LoadFile(dataFile);
    } catch (YAML::Exception& ex) {
      throw std::invalid_argument("Invalid yaml file " + dataFile);
    }

    // parse the colors for the color conversion
    YAML::Node label_remap;
    YAML::Node color_map;
    try {
      label_remap = cfg_data["label_remap"];
      color_map = cfg_data["color_map"];

    } catch (YAML::Exception& ex) {
      std::cerr << "Can't open one of the color dictionaries from "
                << dataFile << ex.what() << std::endl;
    }

    // get the remapping from both dictionaries, in order to speed up conversion
    size_t nrClasses = label_remap.size();
    for (size_t i = 0; i < nrClasses; ++i) {
      cv::Vec3b color = {0, 0, 0};
      labelToBGR.push_back(color);
    }

    YAML::const_iterator it;
    for (it = label_remap.begin(); it != label_remap.end(); ++it) {
      int key = it->first.as<int>();     // <- key
      int label = it->second.as<int>();  // <- label
      cv::Vec3b color = {
          static_cast<uint8_t>(color_map[key][0].as<unsigned int>()),
          static_cast<uint8_t>(color_map[key][1].as<unsigned int>()),
          static_cast<uint8_t>(color_map[key][2].as<unsigned int>())
      };
      labelToBGR[label] = color;
    }
  }

  /* ----------------------------------------------------------------------- */
  // Compute bonnet label accuracy and output results to csv
  void ComputeBonnetAccuracy(const std::string& labelFileName,
                             const std::string& bonnetFileName,
                             const std::string& accuracyFileName) {
    // bonnet has 20 classes
    size_t nrClasses = 20;

    // Cityscapes intensity mapped to bonnet class
    // labelVec[i] = class of pixel having intensity i
    size_t labelVec[34] {19, 19, 19, 19, 19, 19, 19, 0, 1, 19, 19,
      2, 3, 4, 19, 19, 19, 5, 19, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19,
      19, 16, 17, 18};

    // open input and output file
    std::ofstream outputFile;
    outputFile.open(accuracyFileName);
    cv::Mat gtLabel = cv::imread(labelFileName, cv::IMREAD_GRAYSCALE);
    cv::Mat bonnetLabel = cv::imread(bonnetFileName, cv::IMREAD_GRAYSCALE);

    // resize bonnetLabel to inputLabel using the size of gtLabel
    cv::Mat inputLabel;
    cv::resize(bonnetLabel, inputLabel, gtLabel.size(), 0, 0, cv::INTER_NEAREST);

    // initialize matrix P
    // P_ij = the amount of pixels of class i inferred to belong to class j
    gtsam::Matrix P = gtsam::Matrix::Zero(nrClasses, nrClasses);

    for(size_t i=0; i<inputLabel.rows; ++i) {
      for(size_t j=0; j<inputLabel.cols; ++j) {
        // inferred label
        size_t inferredLabel = static_cast<size_t> (inputLabel.at<uint8_t>(i, j));

        // actual label
        uint8_t id = gtLabel.at<uint8_t>(i, j);
        size_t label;
        if(id < 0) {
          label = 19;
        } else {
          label = labelVec[id];
        }

        // add to corresponding entry in P
        P(label, inferredLabel) += 1;
      }
    }

    // write accuracy matrix to outputFile
    for(size_t i=0; i<P.rows(); ++i) {
      for(size_t j=0; j<P.cols()-1; ++j) {
        outputFile << P(i, j) << ",";
      }
      outputFile << P(i, P.cols()) << "\n";
    }

    // close output file
    outputFile.close();
  }

  /* ----------------------------------------------------------------------- */
  // Log ground-truth pixel-level labels for each superpixel
  void LogSuperpixelGTLabels(const std::string& outputFileName,
      const std::string& labelImageName, size_t nrClasses,
      const std::vector<size_t>& labelVec,
      const std::vector<CRFsegmentation::SuperPixel>& superPixels) {
    // open input and output file
    std::ofstream outputFile;
    outputFile.open(outputFileName);
    cv::Mat inputLabel = cv::imread(labelImageName, cv::IMREAD_GRAYSCALE);

    // loop through superpixel
    for(const CRFsegmentation::SuperPixel& superPixel : superPixels) {
      std::vector<int> labelCount;
      for(size_t i=0; i<nrClasses; ++i) { // initialize labelCount
        labelCount.push_back(0);
      }
      // count the number of labels in each superpixel
      for(auto const& pixel: superPixel.pixels) {
        uint8_t gray = inputLabel.at<uint8_t>(pixel.first, pixel.second);
        size_t label;
        if(gray < 0) {
          label = nrClasses - 1;
        } else {
          label = labelVec[gray];
        }
        labelCount[label] += 1;
      }

      // add labels to CSV file
      for(int i=0; i<labelCount.size()-1; ++i) {
        outputFile << labelCount[i] << ",";
      }
      outputFile << labelCount.back() << "\n";
    }

    // close output file
    outputFile.close();
  }

  /* ----------------------------------------------------------------------- */
  // Save labels to designated file
  void SaveLabels(const std::string& outputFileName,
      const std::vector<size_t>& labels) {
    // open output file
    std::ofstream labelFile;
    labelFile.open(outputFileName);
    for(auto& i : labels) {
      labelFile << i << "\n";
    }

    // close output file
    labelFile.close();
  }

  /* ----------------------------------------------------------------------- */
  // Save matrix to designated file
  void SaveMatrix(const std::string& outputFileName, const gtsam::Matrix& M) {
    // open output file
    std::ofstream outputFile;
    outputFile.open(outputFileName);
    for (size_t r = 0; r < M.rows(); ++r) {
      for (size_t c = 0; c < M.cols()-1; ++ c) {
        outputFile << M(r, c) << ",";
      }
      outputFile << M(r, M.cols()-1) << "\n";
    }

    // close output file
    outputFile.close();
  }

} // namespace UtilsFuses

#endif /* UTILSFUSES_H_ */
