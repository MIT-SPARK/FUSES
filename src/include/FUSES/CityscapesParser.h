/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * ------------------------------------------------------------------------- */

/**
 * @file   DavisParser.h
 * @brief  Data parser for Cityscapes dataset
 * @author Siyi Hu, Luca Carlone
 */

#ifndef CityscapesParser_H_
#define CityscapesParser_H_

#include <stdlib.h>
#include <iostream>
#include <string>
#include <experimental/optional>
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/core/core.hpp>

using namespace std;

namespace CRFsegmentation {

/*
 * Class storing file locations
 */
class CityscapesParser {
private:
  const string toRgbFolder = "/leftImg8bit_trainvaltest/leftImg8bit/val/";
  const string toLabelFolder = "/gtFine_trainvaltest/gtFine/val/";

  // dataset
  const string datasetName_;

  // master folder path
  const string folderPath_;

  // rgb image folder path
  vector<string> rgbFolderPath_;

  // label image folder path
  vector<string> labelFolderPath_;

  // number of images in the first i subfolders
  vector<size_t> nrImages_;

  // rgb image file names
  vector<string> rgbFiles_;

  // for color conversion
  vector<cv::Vec3b> labelToBGR_;

  // semantic label dictionary
  YAML::Node cfg_data_;

  struct path_leaf_string {
    string operator()(const boost::filesystem::directory_entry& entry) const{
      return entry.path().leaf().string();
    }
  };

public:
  /* ---------------------------------------------------------------------- */
  // constructor
  CityscapesParser(const string& folderPath, const string& datasetName,
      const experimental::optional<string>& modelPath = experimental::nullopt) :
    folderPath_(folderPath), datasetName_(datasetName) {
    if (!datasetName.compare("all")) {
      // add frankfurt dataset
      rgbFolderPath_.push_back(folderPath_ + toRgbFolder + "frankfurt/");
      labelFolderPath_.push_back(folderPath_ + toLabelFolder + "frankfurt/");

      read_directory(rgbFolderPath_.back(), rgbFiles_);
      nrImages_.push_back(rgbFiles_.size());

      // add lindau dataset
      rgbFolderPath_.push_back(folderPath_ + toRgbFolder + "lindau/");
      labelFolderPath_.push_back(folderPath_ + toLabelFolder + "lindau/");

      read_directory(rgbFolderPath_.back(), rgbFiles_);
      nrImages_.push_back(rgbFiles_.size());

      // add munster dataset
      rgbFolderPath_.push_back(folderPath_ + toRgbFolder + "munster/");
      labelFolderPath_.push_back(folderPath_ + toLabelFolder + "munster/");

      read_directory(rgbFolderPath_.back(), rgbFiles_);
      nrImages_.push_back(rgbFiles_.size());
    } else {
      if (!datasetName.compare("frankfurt")) {
        rgbFolderPath_.push_back(folderPath_ + toRgbFolder + "frankfurt/");
        labelFolderPath_.push_back(folderPath_ + toLabelFolder + "frankfurt/");
      } else if (!datasetName.compare("lindau")) {
        rgbFolderPath_.push_back(folderPath_ + toRgbFolder + "lindau/");
        labelFolderPath_.push_back(folderPath_ + toLabelFolder + "lindau/");
      } else if (!datasetName.compare("munster")){
        rgbFolderPath_.push_back(folderPath_ + toRgbFolder + "munster/");
        labelFolderPath_.push_back(folderPath_ + toLabelFolder + "munster/");
      } else {
        rgbFolderPath_.push_back(folderPath_ + toRgbFolder + "hard/");
        labelFolderPath_.push_back(folderPath_ + toLabelFolder + "hard/");
      }
      // store all rgb image names in rgbFiles_ vector
      read_directory(rgbFolderPath_[0], rgbFiles_);
      nrImages_.push_back(rgbFiles_.size());
    }

    // initialize color if a pretrained model is given
    if (modelPath) {
      string dataFile = *modelPath + "/data.yaml";
      try {
        cfg_data_ = YAML::LoadFile(*modelPath + "/data.yaml");
      } catch (YAML::Exception& ex) {
        throw std::invalid_argument("Invalid yaml file " + dataFile);
      }

      init_color();
    }
  }

  /* ---------------------------------------------------------------------- */
  // store the name of all files in a given directory in a vector of strings
  void read_directory(const string& name, vector<string>& v) {
    boost::filesystem::path p(name);
    boost::filesystem::directory_iterator start(p);
    boost::filesystem::directory_iterator end;
    transform(start, end, back_inserter(v), path_leaf_string());
  }

  /* ---------------------------------------------------------------------- */
  // initialize labelToBGR_ vector
  void init_color() {
    // parse the colors for the color conversion
    YAML::Node label_remap;
    YAML::Node color_map;
    try {
      label_remap = cfg_data_["label_remap"];
      color_map = cfg_data_["color_map"];

    } catch (YAML::Exception& ex) {
      std::cerr << "Can't open one of the color dictionaries from data.yaml."
                << ex.what() << std::endl;
    }

    // get the remapping from both dictionaries, in order to speed up conversion
    int nrClasses = label_remap.size();
    for (int i = 0; i < nrClasses; ++i) {
      cv::Vec3b color = {0, 0, 0};
      labelToBGR_.push_back(color);
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
      labelToBGR_[label] = color;
    }
  }

  /* ---------------------------------------------------------------------- */
  // get an rgb image file name given an index
  void getImage(size_t index, string& filename) const {
    size_t f_index = 0;
    while (f_index < nrImages_.size()) {
      if (index < nrImages_[f_index]) {
        filename = rgbFolderPath_[f_index] + rgbFiles_[index];
        break;
      } else {
        ++f_index;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  // get a label file name given an index
  void getLabel(size_t index, string& filename) const {
    size_t f_index = 0;
    while (f_index < nrImages_.size()) {
      if (index < nrImages_[f_index]) {
        filename = labelFolderPath_[f_index] + rgbFiles_[index];
        break;
      } else {
        ++f_index;
      }
    }

    if(cfg_data_) {
      // replace *rgb* by *semantic*
      filename.replace(filename.end()-15, filename.end()-4,
          "gtFine_labelIds");
    } else {
      // replace *rgb* by *semantic*
      filename.replace(filename.end()-15, filename.end()-4, "gtFine_color");
    }
  }

  /* ---------------------------------------------------------------------- */
  // get a color vector
  const std::vector<cv::Vec3b>& getColorVec() const {
    return labelToBGR_;
  }

  /* ---------------------------------------------------------------------- */
  // get nrClasses from size of labelToBGR_
  int nrClasses() const {
    return labelToBGR_.size();
  }

  /* ---------------------------------------------------------------------- */
  // get total nrImages
  const size_t nrImages() const {
    return std::accumulate(nrImages_.begin(), nrImages_.end(), 0);
  }

};

} // namespace CRFsegmentation

#endif /* CityscapesParser_H_ */
