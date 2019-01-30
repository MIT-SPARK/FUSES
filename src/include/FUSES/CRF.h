/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   CRF.h
 * @brief  Class modeling a conditional random field
 * @author Siyi Hu, Luca Carlone
 */

#ifndef CRF_H_
#define CRF_H_

#include <stdlib.h>

#include "FUSES/UtilsGM.h"

namespace CRFsegmentation {

struct BinaryFactor {
  size_t firstNode;
  size_t secondNode;
  double weight;

  bool equals(const BinaryFactor& rhs, double tol=1e-9) const {
    if((firstNode==rhs.firstNode && secondNode==rhs.secondNode) ||
        (firstNode==rhs.secondNode && secondNode==rhs.firstNode)) {
      if(fabs(weight - rhs.weight) < tol) {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
};

struct UnaryFactor {
  size_t node;
  size_t label;
  double weight = 1;

  bool equals(const UnaryFactor& rhs, double tol=1e-9) const {
    if(node==rhs.node && label==rhs.label) {
      if(fabs(weight - rhs.weight) < tol) {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
};

/* This lightweight class stores a conditional random field with CRF.
 * The binary terms stored should follow the convention that
 * firstNode < secondNode to avoid duplicates.
 * (i.e. edge(i,j) is the same as edge(j,i)) */
class CRF {

public:
  size_t nrNodes = 0;
  size_t nrClasses = 0;
  std::vector<UnaryFactor> unaryFactors;
  std::vector<BinaryFactor> binaryFactors;

  /*----------------------------------------------------------------------------
   * Plain CRF cost: sum_i delta_i(x_i != bar_x_i) + sum_ij delta_ij(x_i != x_j)
   * using this cost, the minimum is zero
   */
  double evaluateMRF(const std::vector<size_t> labels) const {
    if(labels.size() != nrNodes) {
      std::cout <<  "Could not evaluate labeling vector" << std::endl;
      throw std::runtime_error("The size of input vector does not match the "
          "number of variables in the factor graph");
    }
    double cost = 0.0;
    for(size_t i = 0; i < unaryFactors.size(); i++){
      if(labels.at(unaryFactors[i].node) != unaryFactors[i].label){
        cost = cost + unaryFactors[i].weight;
      }
    }
    for(size_t k = 0; k < binaryFactors.size(); k++){
      int firstNode = binaryFactors.at(k).firstNode;
      int secondNode = binaryFactors.at(k).secondNode;
      double weight = binaryFactors.at(k).weight;
      if(labels.at(firstNode) != labels.at(secondNode)){
        cost = cost + 2*weight;
      }
    }
    return cost;
  }

  /*----------------------------------------------------------------------------
   * Cost adopted in FUSES and other approaches:
   * using this cost, the minimum is sum_i delta_i + sum_ij delta_ij
   */
  // TODO LC: this assumes positive weights!
  double evaluate(const std::vector<size_t> labels) const {
    double minCost = 0.0;
    for(size_t i = 0; i < unaryFactors.size(); i++){
      minCost = minCost + unaryFactors[i].weight;
    }
    for(size_t k = 0; k < binaryFactors.size(); k++){
      minCost = minCost + 2*binaryFactors.at(k).weight;
    }
    return evaluateMRF(labels) - minCost;
  }

  double evaluate(const gtsam::Matrix& labelingMatrix) const {
    std::vector<size_t> labeling = UtilsGM::GetLabelsFromMatrix(labelingMatrix,
        nrNodes, nrClasses);
    return evaluate(labeling);
  }

  /* -----------------------------------------------------------------------
   * Reduce the number of classes in CRF, by setting every extra class to the
   * top class (we keep classes 0:targetNrClasses-2 and set remaining to
   * targetNrClasses-1)
  */
  void reduceNrClassesCRF(const size_t targetNrClasses){
    //CRFsegmentation::CRF reducedCRF = fullCrf;
    nrClasses = targetNrClasses;
    // we keep classes 0 to targetNrClasses-2 and set remaining to
    // targetNrClasses-1
    // note: this only affects the unary factors
    for(size_t i = 0; i < unaryFactors.size(); i++){
      if(unaryFactors.at(i).label > targetNrClasses-1)
        unaryFactors.at(i).label = targetNrClasses-1;
    }
  }

  /* -----------------------------------------------------------------------
   * Reduce the number of classes in CRF, by setting every class to
   * newClass = class - (class/targetNrClasses) * targetNrClasses
  */
  void reduceNrClassesCRF2(const size_t targetNrClasses){
    //CRFsegmentation::CRF reducedCRF = fullCrf;
    nrClasses = targetNrClasses;
    // newClass = class - (class/targetNrClasses) * targetNrClasses
    for(size_t i = 0; i < unaryFactors.size(); i++){
      size_t initialClass = unaryFactors.at(i).label;
      size_t newClass = initialClass -
          (initialClass/targetNrClasses) * targetNrClasses;
      unaryFactors.at(i).label = newClass;
    }
  }

  /*----------------------------------------------------------------------------*/
  void print() const {
    std::cout << "\n$$$$$$$$$ Printing CRF $$$$$$$$$" << std::endl;
    for(auto const& unaryFactor : unaryFactors){
      std::cout << "Label of node " << unaryFactor.node << " is "
          << unaryFactor.label << " with weight " << unaryFactor.weight
          << std::endl;
    }

    for(auto const& binaryFactor : binaryFactors){
      std::cout << "Weight between node " << binaryFactor.firstNode
          << " and node " << binaryFactor.secondNode
          << " is " << binaryFactor.weight << std::endl;
    }
    std::cout << "Size of unaryFactor vector = " << unaryFactors.size()
        << std::endl;
    std::cout << "Size of binaryFactor vector = " << binaryFactors.size()
        << std::endl;
    std::cout << "Number of nodes: " << nrNodes << std::endl;
    std::cout << "Number of classes: " << nrClasses << std::endl;
  }

  /*----------------------------------------------------------------------------*/
  bool equals(const CRF& rhs, double tol=1e-9) const {
    // check number of classes
    if(nrClasses != rhs.nrClasses) {
      std::cout << "number of classes not the same: "
          << nrClasses << " != " << rhs.nrClasses << std::endl;
      return false;
    }

    // check number of nodes
    if(nrNodes != rhs.nrNodes) {
      std::cout << "number of nodes not the same: "
          << nrNodes << " != " << rhs.nrNodes << std::endl;
      return false;
    }

    // check unaryFactors
    if(unaryFactors.size() != rhs.unaryFactors.size()) {
      return false;
    } else {
      for(auto& unaryFactor : unaryFactors) {
        auto it = rhs.unaryFactors.begin();
        while(it != rhs.unaryFactors.end()) {
          if (unaryFactor.equals(*it)) {
            break;
          }
          ++it;
        }
        if(it == rhs.unaryFactors.end()) {
          std::cout << "unaryFactor not found!" << std::endl;
          std::cout << "node = " << unaryFactor.node << std::endl;
          return false;
        }
      }
    }

    // check binaryFactors
    if(binaryFactors.size() != rhs.binaryFactors.size()) {
      return false;
    } else {
      // look for each binaryFactor in rhs.binaryFactors
      for(auto& binaryFactor : binaryFactors) {
        auto it = rhs.binaryFactors.begin();
        while(it != rhs.binaryFactors.end()) {
          if (binaryFactor.equals(*it, tol)) {
            break;
          }
          ++it;
        }
        if(it == rhs.binaryFactors.end()) {
          std::cout << "binaryFactor not found! "
              << "first node = " << binaryFactor.firstNode
              << " second node = " << binaryFactor.secondNode
              << " weight = " << binaryFactor.weight << std::endl;
          return false;
        }
      }
    }
    return true;
  }
};

} // namespace CRFsegmentation
#endif /* CRF_H_ */


