/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * ------------------------------------------------------------------------- */

/**
 * @file   Stanford3DSemanticParser.h
 * @brief  Data parser for Standford 2D-3D-Semantics dataset
 * @author Siyi Hu, Luca Carlone
 */

#ifndef Stanford3DSemanticParser_H_
#define Stanford3DSemanticParser_H_

#include <stdlib.h>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>

namespace CRFsegmentation {

/*
 * Class storing file locations
 */
class Stanford3DSemanticParser {
private:
  const std::string toRgbFolder = "/rgb/";
  const std::string toLabelFolder = "/semantic/";

  struct path_leaf_string {
    std::string operator()(const boost::filesystem::directory_entry& entry) const{
      return entry.path().leaf().string();
    }
  };

public:
  // master folder path
  const std::string folderPath_;

  // rgb image folder path
  const std::string rgbFolderPath_;

  // label image folder path
  const std::string labelFolderPath_;

  // rgb image file names
  std::vector<std::string> rgbFiles_;

  /* ---------------------------------------------------------------------- */
  // constructor
  Stanford3DSemanticParser(const std::string& folderPath) :
    folderPath_(folderPath), rgbFolderPath_(folderPath_ + toRgbFolder),
    labelFolderPath_(folderPath_ + toLabelFolder) {

    // store all rgb image names in rgbFiles_ vector
    read_directory(rgbFolderPath_, rgbFiles_);
  }

  /* ---------------------------------------------------------------------- */
  // store the name of all files in a given directory in a vector of strings
  void read_directory(const std::string& name, std::vector<std::string>& v) {
    boost::filesystem::path p(name);
    boost::filesystem::directory_iterator start(p);
    boost::filesystem::directory_iterator end;
    std::transform(start, end, back_inserter(v), path_leaf_string());
  }

  /* ---------------------------------------------------------------------- */
  // get an rgb image file name given an index
  void getImage(size_t index, std::string& filename) const {
    filename = rgbFolderPath_ + rgbFiles_[index];
  }

  /* ---------------------------------------------------------------------- */
  // get a label file name given an index
  void getLabel(size_t index, std::string& filename) const {
    filename = labelFolderPath_ + rgbFiles_[index];
    // replace *rgb* by *semantic*
    filename.replace(filename.end()-7, filename.end()-4, "semantic");
  }
};

} // namespace CRFsegmentation

#endif /* Stanford3DSemanticParser_H_ */
