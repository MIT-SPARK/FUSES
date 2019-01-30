/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   UtilsGM.h
 * @brief  Utilities for graphical models
 * @author Luca Carlone, Siyi Hu
 */

#ifndef UTILSGM_H_
#define UTILSGM_H_

#include <stdlib.h>
#include <opengv/point_cloud/methods.hpp>
#include <gtsam/base/Matrix.h>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sys/time.h>

class UtilsGM {

public:
  /* ----------------------------------------------------------------------- */
  // This function essentially reshapes matrix
  static gtsam::Matrix Reshape(const gtsam::Matrix& v, const size_t nrRows,
      const size_t nrCols) {
    if (v.rows() * v.cols() != nrRows * nrCols) {
      std::cout <<  "reshape error "  << std::endl;
      throw std::runtime_error("nr of elements in the matrices does not match.");
    }
    gtsam::Matrix v_reshaped = gtsam::Matrix::Zero(nrRows,nrCols);
    size_t ind = 0;
    for(size_t i = 0; i<nrRows * nrCols; i++){
      v_reshaped( i % nrRows , i / nrRows ) = v( i % v.rows() , i / v.rows() );
      ind++;
    }
    return v_reshaped;
  }

  /* ----------------------------------------------------------------------- */
  static std::vector<size_t> GetLabelsFromMatrix(
      const gtsam::Matrix& labelingMatrix, size_t nrNodes, size_t nrClasses){
    // check dimensions of labelingMatrix
    if(labelingMatrix.rows() != nrNodes) {
      std::cout <<  "Could not evaluate labeling matrix" << std::endl;
      throw std::runtime_error("The number of rows in the input matrix does "
          "not match the number of variables in the factor graph");
    }
    if(labelingMatrix.cols() != nrClasses) {
      std::cout <<  "Could not evaluate labeling matrix" << std::endl;
      throw std::runtime_error("The number of columns in the input matrix does "
          "not match the variable space in the factor graph");
    }
    for(size_t r=0; r<labelingMatrix.rows();r++){
      for(size_t c=0; c<labelingMatrix.cols();c++){
        if(labelingMatrix(r,c)!=0 && labelingMatrix(r,c)!=1)
          throw std::runtime_error("The input matrix is not binary");
      }
    }
    // sum of rows should be 1
    gtsam::Vector X1 = labelingMatrix * gtsam::Vector::Ones(nrClasses);
    for(size_t r=0; r<X1.size(); r++){
      if(std::abs(X1(r)-1) > 1e-5)
        throw std::runtime_error("The input matrix rows do not sum up to 1");
    }
    // find the location of the maximum number in each row and add index to
    // labeling vector
    std::vector<size_t> labeling;
    gtsam::Matrix::Index maxIndex;

    for(size_t i=0; i<nrNodes; ++i) {
      labelingMatrix.row(i).maxCoeff(&maxIndex);
      labeling.push_back((size_t)maxIndex);
    }
    return labeling;

    /* Alternative approach:
     *   Matrix colVec(k, 1);
          for(int i=0; i<k; ++i) {
            colVec(i, 0) = i;
          }
          Matrix temp = resultMatrix * colVec;

          vector<int> resultVec;
          for(int i=0; i<n; ++i) {
            resultVec.push_back( (int)round(temp(i, 0)) );
          }
     */
  }

  /* ----------------------------------------------------------------------- */
  static gtsam::Matrix GetMatrixFromLabels(const std::vector<size_t>& labeling,
      size_t nrNodes, size_t nrClasses){
    if(labeling.size() != nrNodes) {
      std::cout <<  "Could not evaluate labeling matrix" << std::endl;
      throw std::runtime_error("The number of entries in the label vector "
          "not match the number of variables in the factor graph");
    }
    gtsam::Matrix X = gtsam::Matrix::Zero(nrNodes,nrClasses);

    // put entries equal to 1 for each node label
    for(size_t i=0; i<nrNodes;i++){
      size_t l = labeling.at(i);
      if(l > nrClasses-1){
        throw std::runtime_error("GetMatrixFromLabels: Label i exceeds number "
            "of labels");
      }
      X(i,l) = 1.0;
    }
    return X;
  }

  /* ----------------------------------------------------------------------------- */
  //  print standard vector with header:
  template< typename T >
  static void PrintVector(const std::vector<T> vect,const std::string vectorName){
    std::cout << vectorName << std::endl;
    for(auto si : vect) std::cout << " " << si;
    std::cout << std::endl;
  }

};
#endif /* UTILSGM_H_ */
