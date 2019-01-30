/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   FusesDA.h
 * @brief  Fast Understanding Via Semidefinite Segmentation
 *         This version implements a dual ascent strategy to enforce that each node has a single label
 * @author Siyi Hu, Luca Carlone
 */

#ifndef FUSESDA_H_
#define FUSESDA_H_

#include "FUSES/Fuses.h"

namespace FUSES {

/* ============================= FUSES =========================== */
class FusesDA : public Fuses {
 public:
  double alpha_init = 0.05; // initial stepsize

  unsigned int max_iterations_DA = 100;

  double min_eqViolation_tol = 0.1;

  gtsam::Vector y_k; // vector of dual variables

  gtsam::Vector eqViolation;

  double norm_constraint_violation; // violation of eq constraint X 1 = 1

  /** FusesDA constructor: taking crf and options (optional)
   * Will generate a random initial guess */
  FusesDA(const CRFsegmentation::CRF& crf, const boost::optional<FUSESOpts&>
    options = boost::none) : Fuses(crf,options) {
    y_k = gtsam::Vector::Zero(crf.nrNodes);
    eqViolation = gtsam::Vector::Zero(crf.nrNodes);
    norm_constraint_violation = 0.0;
  }

  /** FusesDA constructor: taking initial guess in matrix form (only for expert
   * users, will set r0 according to Y0) and options (optional) */
  FusesDA(const CRFsegmentation::CRF& crf, const gtsam::Matrix &Y0,
      const boost::optional<FUSESOpts&> options = boost::none) : Fuses(crf,Y0,options) {
    y_k = gtsam::Vector::Zero(crf.nrNodes);
    eqViolation = gtsam::Vector::Zero(crf.nrNodes);
    norm_constraint_violation = 0.0;
  }

  /** FusesDA constructor: taking initial guess in vector form nodeLabels[i]
   * corresponds to class of node i and options (optional) */
  FusesDA(const CRFsegmentation::CRF& crf, const std::vector<size_t>& nodeLabels,
      const boost::optional<FUSESOpts&> options = boost::none): Fuses(crf,nodeLabels,options) {
    y_k = gtsam::Vector::Zero(crf.nrNodes);
    eqViolation = gtsam::Vector::Zero(crf.nrNodes);
    norm_constraint_violation = 0.0;
  }

  /** Helper function computing the offset trace(topRightBlock(1_K*yk')*YY') */
  double computeOffset(const gtsam::Matrix& yk, const gtsam::Matrix& Yk) const {
    // TODO: this might be further improved
    double offset = ( Yk.topRows(crf_.nrNodes).transpose()
        * (yk * gtsam::Matrix::Ones(1,crf_.nrClasses))
        * Yk.bottomRows(crf_.nrClasses) ).trace();

    // Full (but more expensive) math:
    //    gtsam::Matrix Q_y_k =
    //        gtsam::Matrix::Zero(crf_.nrNodes+crf_.nrClasses,crf_.nrNodes+crf_.nrClasses);
    //    Q_y_k.block(0,crf_.nrNodes,crf_.nrNodes,crf_.nrClasses) =
    //        yk * gtsam::Matrix::Ones(1,crf_.nrClasses);
    //    double offset = (Yk.transpose() * Q_y_k *Yk).trace();
    return offset;
  }

  // TODO: check destructor
  //  /** Fuses destructor */
  //  ~FusesDA();

  void solveDA(){
    // constants:
    gtsam::Vector ones_N = gtsam::Vector::Ones(crf_.nrNodes);
    gtsam::Vector ones_K = gtsam::Vector::Ones(crf_.nrClasses);
    Q_.reserve(2*crf_.binaryFactors.size() + 2*crf_.nrNodes + crf_.nrClasses);

    // Store original upper right block of Q_:
    gtsam::Matrix G_const = gtsam::Matrix::Zero(crf_.nrNodes,crf_.nrClasses);
    for(auto const& unaryFactor : crf_.unaryFactors){
      G_const(unaryFactor.node, unaryFactor.label) = -0.5; // 1/2 G
      // diag[n + unaryFactor.label] += 0.5; // TODO LC: check preconditioning
    }

    // check initial violation
    eqViolation = Y_.topRows(crf_.nrNodes) *
              (Y_.bottomRows(crf_.nrClasses).transpose() * ones_K) - ones_N;
    norm_constraint_violation = eqViolation.norm(); // 2 norm
    std::cout << "initial constraint_violation: " << norm_constraint_violation << std::endl;

    // initialize Dual variables
    y_k = gtsam::Vector::Zero(crf_.nrNodes);

    for(size_t iterDA = 0; iterDA < max_iterations_DA; iterDA++){
      // update Lagrangian via Q_
      for(size_t r=0; r<crf_.nrNodes;r++){
        for(size_t c=0; c<crf_.nrClasses;c++){
          Q_.coeffRef(r, crf_.nrNodes + c) = G_const(r,c) + 0.5*y_k(r);
        }
      }

      // the number of iterates from previous iterations
      // result_.function_values has size equal to the number of DA iteration multiplied by the
      // staircase steps in each iteration
      size_t nrPrevStairCase = result_.function_values.size();

      // solve Lagrangian minimization
      solve(); // calls FUSES on the Lagrangian

      Y_ = result_.Yopt; // warm start next iteration

      // remove offset from function_values
      size_t nrCurrentStairCase = result_.function_values.size();
      if(options_.log_iterates){
        for(size_t stairCase=nrPrevStairCase; stairCase<nrCurrentStairCase; ++stairCase) {
          if(result_.function_values[stairCase].size() != result_.iterates[stairCase].size()){
            std::cout << "a!=b -> " << result_.function_values[stairCase].size() <<
                " " << result_.iterates[stairCase].size() << std::endl;
            // throw std::runtime_error("FusesDA: incorrect size of logged info (2)");
            if(result_.iterates[stairCase].size() > 0){
              std::cout << "LC TO CHECK: ADDING COPY OF ITERATE (1)" << std::endl;
              result_.iterates[stairCase].push_back(result_.iterates[stairCase].back());
            }else{
              std::cout << "LC TO CHECK: ADDING COPY OF ITERATE (2)" << std::endl;
              result_.iterates[stairCase].push_back(Y_); // TODO: sometimes iterates is empty
            }
          }
          if(result_.function_values[stairCase].size() != result_.iterates[stairCase].size()){
            std::cout << "nr fvalues = " << result_.function_values[stairCase].size() << std::endl;;
            std::cout << "nr iterates = " << result_.iterates[stairCase].size() << std::endl;;
            throw std::runtime_error("FusesDA: incorrect size of logged info (1)");
          }
          // correct the latest objectives
          for(size_t i=0; i<result_.function_values[stairCase].size(); ++ i) { // nr of objectives
            result_.function_values[stairCase][i] -= computeOffset(y_k, result_.iterates[stairCase][i]);
          }
        }
      }else{
        std::cout << "FusesDA: cannot correct cost!! " << std::endl;;
        for(size_t stairCase=nrPrevStairCase; stairCase<nrCurrentStairCase; ++stairCase) {
          // in absence of iterates, we set the objective to zero
          for(size_t i=0; i<result_.function_values[stairCase].size(); ++ i) { // nr of objectives
            result_.function_values[stairCase][i] = 0;
          }
        }
      }

      // update Dual variables
      // gtsam::Matrix Z_ur = result_.Yopt.topRows(crf_.nrNodes) * result_.Yopt.bottomRows(crf_.nrClasses).transpose();
      eqViolation = result_.Yopt.topRows(crf_.nrNodes) *
          (result_.Yopt.bottomRows(crf_.nrClasses).transpose() * ones_K) - ones_N;

      norm_constraint_violation = eqViolation.norm(); // 2 norm
      if(norm_constraint_violation < min_eqViolation_tol){
        break; // stop itertions
      }

      // dual update
      double alpha = alpha_init / sqrt(double(iterDA)+1.0);
      y_k = y_k + alpha*(eqViolation);

      std::cout << "\n DA iteration nr: " << iterDA << std::endl;
      // std::cout << "current optimal value: " << result_.SDPval << std::endl;
      // std::cout << "alpha: " << alpha << std::endl;
      std::cout << "constraint_violation: " << norm_constraint_violation << std::endl;
      std::cout << "current size of Y: (" << result_.Yopt.rows() << " x " << result_.Yopt.cols() << ")"<< std::endl;
      // result_.timing.print();
    }
    // finally compute the objective without contribution of dual variables
    result_.Fxhat = crf_.evaluate(result_.xhat);
  }
}; // Class FusesDA

} // namespace FUSES

#endif /* FUSESDA_H_ */
