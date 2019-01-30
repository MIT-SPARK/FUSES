/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * ------------------------------------------------------------------------- */

/**
 * @file   OpenGMParser.h
 * @brief  Parsing CRF to factor graph in opengm
 * @author Siyi Hu, Luca Carlone
 */

#ifndef OpenGMParser_H_
#define OpenGMParser_H_

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/graphcut.hxx>                     // alpha-Expansion
#include <opengm/inference/alphaexpansion.hxx>               // alpha-Expansion
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx> // alpha-Expansion
#include <opengm/inference/messagepassing/messagepassing.hxx>  // LBP
#include <opengm/inference/external/trws.hxx> // TRW-S
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <Eigen/Dense>
#include <gtsam/base/Lie.h>
#include <fstream>

#include "FUSES/CRF.h"
#include "FUSES/UtilsGM.h"

namespace CRFsegmentation {

class OpenGMParser {
private:
  // type definitions of openGM
  typedef opengm::SimpleDiscreteSpace<> Space;
  typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(
      opengm::ExplicitFunction<double> , opengm::PottsFunction<double>),
      Space> Model;

  // cost when labels are the different
  const double diffLabel = 0.0;

  // Graphical Model
  Model* gm_ = nullptr;

  // number of labels per variable
  size_t nrClasses_;

  // number of variables
  size_t nrNodes_;

  // initial starting point (will not be used if size is 0)
  std::vector<size_t> initialLabels_;

public:

  /* ---------------------------------------------------------------------- */
  // constructor: construct graphical model from crf
  OpenGMParser(const CRF& crf) : nrClasses_(crf.nrClasses),
      nrNodes_(crf.nrNodes){

    // Initialize graphical model
    size_t shape[] = {nrClasses_};
    Space space(nrNodes_, nrClasses_);
    gm_ = new Model(space);

    // add unary factors
    for (auto& unaryFactor : crf.unaryFactors) {
      opengm::ExplicitFunction<double> f(shape, shape + 1, diffLabel);
      f(unaryFactor.label) = -unaryFactor.weight;
      Model::FunctionIdentifier fid = gm_->addFunction(f);
      size_t v[] = {(size_t)unaryFactor.node};
      gm_->addFactor(fid, v, v + 1);
    }

    // add binary factors
    for (auto& binaryFactor : crf.binaryFactors) {
      opengm::PottsFunction<double> f(nrClasses_, nrClasses_,
          -2*binaryFactor.weight, diffLabel);
          // multiply by 2 because crf only store half of BFs
      Model::FunctionIdentifier fid = gm_->addFunction(f);
      size_t v[] = {(size_t)binaryFactor.firstNode,
          (size_t)binaryFactor.secondNode};
      gm_->addFactor(fid, v, v + 2);
    }
  }

  /* ---------------------------------------------------------------------- */
  // constructor: construct from HDF5 file
  OpenGMParser(const std::string& fileName) {
    // load graphical model
    gm_ = new Model();
    opengm::hdf5::load(*gm_, fileName, "gm");

    // assume all variables have the same number of labels
    nrClasses_ = gm_->numberOfLabels(0);
    nrNodes_ = gm_->numberOfVariables();
  }

  /* ---------------------------------------------------------------------- */
  // destructor
  ~OpenGMParser() {
    if(gm_)
      delete gm_;
  }

  /* ---------------------------------------------------------------------- */
  // accessory function
  const size_t& nrNodes() const {return nrNodes_;}
  const size_t& nrClasses() const {return nrClasses_;}

  /* ---------------------------------------------------------------------- */
  // set initial starting point
  template<typename T>
  void setInitialLabels(std::vector<T>& labels) {
    std::copy( labels.begin(), labels.end(), initialLabels_ );
  }

  /* ---------------------------------------------------------------------- */
  // compute results using Loopy Belief Propagation
  // (setStartingPoint not implemented in openGM)
  void computeLBP(std::vector<size_t>& labeling,
      std::vector<double>& iterations, std::vector<double>& values,
      std::vector<double>& times, size_t maxIter=100,
      double convergenceTol=1e-7, double damping=0, bool verbose=false)
      const {
    typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Minimizer>
        UpdateRules;
    typedef opengm::MessagePassing<Model, opengm::Minimizer, UpdateRules,
        opengm::MaxDistance> BeliefPropagation;

    BeliefPropagation::Parameter parameter(maxIter, convergenceTol, damping);
    BeliefPropagation bp(*gm_, parameter);

    // optimize (approximately)
    // (first two parameters in visitor are set to default values in OpenGM)
    BeliefPropagation::TimingVisitorType visitor(1, 0, verbose);
    bp.infer(visitor);
    iterations = visitor.getIterations();
    values = visitor.getValues();
    times = visitor.getTimes();

    // obtain the (approximate) argmin
    bp.arg(labeling);
  }

  /* ---------------------------------------------------------------------- */
  // compute results using alpha-Expansion
  void computeAE(std::vector<size_t>& labeling,
      std::vector<double>& iterations, std::vector<double>& values,
      std::vector<double>& times, bool verbose=false) const {
    typedef opengm::external::MinSTCutKolmogorov<size_t, double> MinStCutType;
    typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType>
        MinGraphCut;
    typedef opengm::AlphaExpansion<Model, MinGraphCut> MinAlphaExpansion;

    MinAlphaExpansion ae(*gm_);

    if (initialLabels_.size() > 0) {
      ae.setStartingPoint(initialLabels_.begin());
      std::cout << "Using specified initial labels." << std::endl;
    }

    // optimize (approximately)
    // (first two parameters in visitor are set to default values in OpenGM)
    MinAlphaExpansion::TimingVisitorType visitor(1, 0, verbose);
    ae.infer(visitor);
    iterations = visitor.getIterations();
    values = visitor.getValues();
    times = visitor.getTimes();

    // obtain the (approximate) argmin
    ae.arg(labeling);
  }

  /* ---------------------------------------------------------------------- */
  // compute results using tree-reweighted message passing (TRW-S)
  // (setStartingPoint not implemented in openGM)
  void computeTRWS(std::vector<size_t>& labeling,
      std::vector<double>& iterations, std::vector<double>& values,
      std::vector<double>& times, bool verbose=false) const {
    typedef opengm::external::TRWS<Model> TRWS;

    TRWS trws(*gm_);

    // optimize (approximately)
    // (first two parameters in visitor are set to default values in OpenGM)
    TRWS::TimingVisitorType visitor(1, 0, verbose);
    trws.infer(visitor);
    iterations = visitor.getIterations();
    values = visitor.getValues();
    times = visitor.getTimes();

    // obtain the (approximate) argmin
    trws.arg(labeling);
  }

  /* ---------------------------------------------------------------------- */
  // save model from hdf5 file
  void saveModel(const std::string& outputFile) const {
    opengm::hdf5::save(*gm_, outputFile, "gm");
  }

  /* ---------------------------------------------------------------------- */
  // get crf from model
  CRF getCRF() const {
    // initialize crf
    CRF crf;
    crf.nrNodes = nrNodes_;
    crf.nrClasses = nrClasses_;

    for (size_t i=0; i<gm_->numberOfFactors(); ++i) {
      Model::IndependentFactorType factor = (*gm_)[i];
      if (factor.numberOfVariables() == 1) {
        UnaryFactor unaryFactor;
        unaryFactor.node = factor.variableIndex(0);
        for (int label=0; label<nrClasses_; ++label) {
          if (factor(label) < -0.01) {
            unaryFactor.label = label;
            unaryFactor.weight = -factor(label);
            crf.unaryFactors.push_back(unaryFactor);
            break;
          }
        }
      } else {
        BinaryFactor binaryFactor;
        binaryFactor.firstNode = factor.variableIndex(0);
        binaryFactor.secondNode = factor.variableIndex(1);
        // weight = (1/2)*(cost_for_same_labeling - cost_for_diff_labeling)
        binaryFactor.weight = (factor(0, 1) - factor(0, 0)) * 0.5;
        crf.binaryFactors.push_back(binaryFactor);
      }
    }
    return crf;
  }

  /* ---------------------------------------------------------------------- */
  // evaluate total cost for a given labeling
  double evaluate(const std::vector<size_t>& labeling) const {
    // check dimensions of labeling
    if (labeling.size() != nrNodes_) {
      std::cout <<  "Could not evaluate labeling vector" << std::endl;
      throw std::runtime_error("The size of input vector does not match the "
          "number of variables in the factor graph");
    }
    return gm_ -> evaluate(labeling);
  }

  double evaluate(const gtsam::Matrix& labelingMatrix) const {
    std::vector<size_t> labeling = UtilsGM::GetLabelsFromMatrix(labelingMatrix,
        nrNodes_, nrClasses_);
    return evaluate(labeling);
  }
};
} // namespace CRFsegmentation

#endif /* OpenGMParser_H_ */
