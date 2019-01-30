/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   Fuses2DA.h
 * @brief  Fast Understanding Via Semidefinite Segmentation
 *         This version implements a dual ascent strategy to enforce that each node has a single label
 * @author Siyi Hu, Luca Carlone
 */

#ifndef FUSES2DA_H_
#define FUSES2DA_H_

#include "FUSES/Fuses2.h"

namespace FUSES {

/* ============================= FUSES =========================== */
class Fuses2DA : public Fuses2 {
public:
  double alpha_init = 0.05; // initial stepsize

  unsigned int max_iterations_DA = 2000;

  double min_eqViolation_tol = 1;

  bool use_constant_stepsize = false;

  SpMatrix Aeq; // matrix of equality constraint (1 label per node)
  gtsam::Vector beq; // vector of equality constraint (1 label per node)

  gtsam::Vector y_k; // vector of dual variables

  gtsam::Vector eqViolation;

  double norm_constraint_violation; // violation of eq constraint Aeq x = beq

  /** Fuses2DA constructor: taking crf and options (optional)
   * Will generate a random initial guess */
  Fuses2DA(const CRFsegmentation::CRF& crf, const boost::optional<FUSESOpts&>
  options = boost::none) : Fuses2(crf,options) {
    y_k = gtsam::Vector::Zero(crf.nrNodes);
    eqViolation = gtsam::Vector::Zero(crf.nrNodes);
    norm_constraint_violation = 0.0;
    buildConstraints();
  }

  /** Fuses2DA constructor: taking initial guess in matrix form (only for expert
   * users, will set r0 according to Y0) and options (optional) */
  Fuses2DA(const CRFsegmentation::CRF& crf, const gtsam::Matrix &Y0,
      const boost::optional<FUSESOpts&> options = boost::none) :
        Fuses2(crf,Y0,options) {
    y_k = gtsam::Vector::Zero(crf.nrNodes);
    eqViolation = gtsam::Vector::Zero(crf.nrNodes);
    norm_constraint_violation = 0.0;
    buildConstraints();
  }

  /** Fuses2DA constructor: taking initial guess in vector form nodeLabels[i]
   * corresponds to class of node i and options (optional) */
  Fuses2DA(const CRFsegmentation::CRF& crf,
      const std::vector<size_t>& nodeLabels,
      const boost::optional<FUSESOpts&> options = boost::none):
        Fuses2(crf, nodeLabels, options) {
    y_k = gtsam::Vector::Zero(crf.nrNodes);
    eqViolation = gtsam::Vector::Zero(crf.nrNodes);
    norm_constraint_violation = 0.0;
    buildConstraints();
  }

  // TODO: check destructor
  //  /** Fuses2 destructor */
  //  ~Fuses2DA();

  /** Helper function computing the offset trace(topRightBlock(1_K*yk')*YY') */
  double computeOffset(const gtsam::Matrix& yk, const gtsam::Matrix& Yk) const {
    // TODO: this might be further improved
    gtsam::Vector g_v = gtsam::Vector::Zero(crf_.nrNodes*crf_.nrClasses);
    g_v = Aeq.transpose() * yk;
    double offset = ( Yk.topRows(crf_.nrNodes*crf_.nrClasses).transpose()
        * g_v * Yk.bottomRows(1) ).trace();

    // Full (but more expensive) math:
    //    gtsam::Matrix Q_y_k =
    //        gtsam::Matrix::Zero(crf_.nrNodes+crf_.nrClasses,crf_.nrNodes+crf_.nrClasses);
    //    Q_y_k.block(0,crf_.nrNodes,crf_.nrNodes,crf_.nrClasses) =
    //        yk * gtsam::Matrix::Ones(1,crf_.nrClasses);
    //    double offset = (Yk.transpose() * Q_y_k *Yk).trace();
    return offset;
  }

  /** Helper function to build linear constraints Aeq x = beq (unique node labeling) */
  void buildConstraints(){
    Aeq.resize(crf_.nrNodes, crf_.nrNodes*crf_.nrClasses);
    Aeq.reserve(crf_.nrNodes*crf_.nrClasses);
    for(size_t i=0; i<crf_.nrNodes ;i++){
      for(size_t j=0; j<crf_.nrClasses ;j++){
        //std::cout << i << " " << i*crf_.nrClasses+j << std::endl;
        Aeq.insert(i,i*crf_.nrClasses+j) = 1.0;
      }
    }
    beq = (2.0-double(crf_.nrClasses)) * gtsam::Vector::Ones(crf_.nrNodes);
  }

  /** solve CRF using dual ascent + Fuses2
   * Input is an optional vector of initial dual variables
   * */
  void solveDA(gtsam::Vector y_k_init = gtsam::Vector::Zero(0)){
    Q_.reserve(crf_.nrClasses * (crf_.binaryFactors.size() + 2*crf_.nrNodes + crf_.nrClasses) + 2*crf_.nrNodes*crf_.nrClasses);

    // Store original upper right block of Q_:
    gtsam::Vector g_const = gtsam::Vector::Zero(crf_.nrNodes*crf_.nrClasses);
    for(size_t i=0; i< crf_.nrNodes*crf_.nrClasses ;i++){
      g_const(i) = 2.0*Q_.coeff(i,crf_.nrNodes*crf_.nrClasses); // take last column from sparse mat
    }

    // initialize Dual variables
    gtsam::Vector g_var = gtsam::Vector::Zero(crf_.nrNodes*crf_.nrClasses); // allocate contribution of dual var to lagrangian

    if(y_k_init.size() == crf_.nrNodes){ // use provide initial guess
      y_k = y_k_init;
    }else{
      std::cout << "-- Using given initial guess for y_k" << std::endl;
      y_k = gtsam::Vector::Zero(crf_.nrNodes);
    }

    // Start dual ascent iterations!
    for(size_t iterDA = 0; iterDA < max_iterations_DA; iterDA++){
      // update Lagrangian via Q_
      g_var = g_const + (Aeq.transpose() * y_k);
      double cumAbsSum = 0.0;
      for(size_t r=0; r<crf_.nrNodes*crf_.nrClasses;r++){
        double entry =  0.5 * g_var(r);
        Q_.coeffRef(r, crf_.nrNodes*crf_.nrClasses) = entry;
        Q_.coeffRef(crf_.nrNodes*crf_.nrClasses, r) = entry;
        cumAbsSum += fabs(entry);
      }

      //      if(options_.precon == Jacobi) {
      //        gtsam::Vector diag = JacobiPrecon_.diagonal();
      //        diag(Q_.rows()-1) = cumAbsSum; // diagonal preconditioning matrix
      //        JacobiPrecon_ = diag.cwiseInverse().asDiagonal(); // diagonal preconditioning matrix
      //      }

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
            // std::cout << "result_.iterates[stairCase] " << result_.iterates[stairCase].size() << std::endl;
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
            throw std::runtime_error("Fuses2DA: incorrect size of logged info (1)");
          }
          // correct the latest objectives
          for(size_t i=0; i<result_.function_values[stairCase].size(); ++ i) { // nr of objectives
            result_.function_values[stairCase][i] -= computeOffset(y_k, result_.iterates[stairCase][i]);
            // std::cout << " " << result_.function_values[stairCase][i];
          }
        }
      }else{
        std::cout << "Fuses2DA: cannot correct cost!! " << std::endl;;
        for(size_t stairCase=nrPrevStairCase; stairCase<nrCurrentStairCase; ++stairCase) {
          // in absence of iterates, we set the objective to zero
          for(size_t i=0; i<result_.function_values[stairCase].size(); ++ i) { // nr of objectives
            result_.function_values[stairCase][i] = 0;
          }
        }
      }

      // update Dual variables
      eqViolation =
          Aeq * (Y_.topRows(crf_.nrNodes*crf_.nrClasses) * Y_.bottomRows(1).transpose()) - beq;

      norm_constraint_violation = eqViolation.norm(); // 2 norm
      if(norm_constraint_violation < min_eqViolation_tol){
        break; // stop itertions
      }

      // dual update
      double alpha;
      if (use_constant_stepsize){
        alpha = alpha_init;
      }else{
        alpha = alpha_init / sqrt(double(iterDA)+1.0);
      }
      y_k = y_k + alpha*(eqViolation);

//       std::cout << "\n DA iteration nr: " << iterDA << std::endl;
//       std::cout << "current optimal value: " << result_.SDPval << std::endl;
//       std::cout << "alpha: " << alpha << std::endl;
//       std::cout << "constraint_violation: " << norm_constraint_violation << std::endl;
//       std::cout << "y_k norm" << y_k.norm() << std::endl;
//       std::cout << "current size of Y: (" << result_.Yopt.rows() << " x " << result_.Yopt.cols() << ")"<< std::endl;
      // result_.timing.print();
    }
    // finally compute the objective without contribution of dual variables

    result_.Fxhat = crf_.evaluate(result_.xhat);
  }
}; // Class Fuses2DA

} // namespace FUSES

#endif /* FUSES2DA_H_ */
