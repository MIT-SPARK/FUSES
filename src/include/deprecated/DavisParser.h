/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * ------------------------------------------------------------------------- */

/**
 * @file   DavisParser.h
 * @brief  Data parser for DAVIS dataset
 * @author Siyi Hu, Luca Carlone
 */

#ifndef DavisParser_H_
#define DavisParser_H_

#include <stdlib.h>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>

using namespace std;


namespace CRFsegmentation {

/*
 * Class storing file locations
 */
class DavisParser {
private:
  const string toRgbFolder = "/JPEGImages/480p/";
  const string toLabelFolder = "/Annotations/480p/";

  struct path_leaf_string {
    string operator()(const boost::filesystem::directory_entry& entry) const{
      return entry.path().leaf().string();
    }
  };

public:
  // dataset name
  const string dataName_;

  // master folder path
  const string folderPath_;

  // rgb image folder path
  const string rgbFolderPath_;

  // label image folder path
  const string labelFolderPath_;

  // rgb image file names
  vector<string> rgbFiles_;

  /* ---------------------------------------------------------------------- */
  // constructor
  DavisParser(const string& folderPath, const string& dataName) :
    folderPath_(folderPath), dataName_(dataName),
    rgbFolderPath_(folderPath_ + toRgbFolder + dataName + "/"),
    labelFolderPath_(folderPath_ + toLabelFolder + dataName + "/") {

    // store all rgb image names in rgbFiles_ vector
    read_directory(rgbFolderPath_, rgbFiles_);
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
  // get an rgb image file name given an index
  void getImage(size_t index, string& filename) const {
    filename = rgbFolderPath_ + rgbFiles_[index];
  }

  /* ---------------------------------------------------------------------- */
  // get a label file name given an index
  void getLabel(size_t index, string& filename) const {
    filename = labelFolderPath_ + rgbFiles_[index];
    // replace *rgb* by *semantic*
    filename.replace(filename.end()-3, filename.end(), "png");
  }
};

} // namespace CRFsegmentation

#endif /* DavisParser_H_ */
