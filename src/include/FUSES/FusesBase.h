/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   Fuses.h
 * @brief  Fast Understanding Via Semidefinite Segmentation
 * @author Siyi Hu, Luca Carlone, (based on code from David Rosen)
 */

#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
#include <SymEigsSolver.h>
#include <MatOp/SparseSymMatProd.h>
#include <gtsam/base/Lie.h>
#include <boost/optional.hpp>
#include <boost/algorithm/string.hpp>

#include "FUSES/CRF.h"
#include "FUSES/Timer.h"
#include "FUSES/UtilsGM.h"
#include "Optimization/Smooth/TNT.h"

namespace FUSES {

/* ================== INITIAL DEFINITIONS =========================== */
typedef Eigen::SparseMatrix<double> SpMatrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagMatrix;

/** The type of the incomplete Cholesky decomposition we will use for
 * preconditioning the conjugate gradient iterations in the RTR method */
typedef Eigen::IncompleteCholesky<double> IncompleteCholeskyFactorization;

/** The set of available preconditioning strategies to use in the Riemannian
 * Trust Region when solving this problem */
enum Preconditioner { None, Jacobi, IncompleteCholesky };

/** The set of available rounding strategies to round solution matrix to
 * a binary matrix */
enum Rounding { WinnerTakeAll, RandomizedProj, Sampling };

/** These enumerations describe the termination status of the Fuses algorithm */
enum FUSESStatus {
  /** The algorithm converged to a certified global optimum */
  GLOBAL_OPT,
  /** The algorithm converged to a saddle point, but the backtracking line
     search was unable to escape it */
  SADDLE_POINT,
  /** The algorithm converged to a first-order critical point, but the
     minimum-eigenvalue computation did not converge to sufficient precision
     to enable its characterization */
  EIG_IMPRECISION,
  /** The algorithm exhausted the maximum number of iterations of the
     Riemannian Staircase before finding an optimal solution */
  RS_ITER_LIMIT,
  /** The algorithm exhausted the allotted total computation time before
     finding an optimal solution */
  ELAPSED_TIME
};

/* ================== FUSES: OPTIONS struct =========================== */
/** This struct contains the various parameters that control the FUSES
 * algorithm */
// types of initialization in Fuses
enum InitializationType { FromUnary, RandomClasses, Random };

struct FUSESOpts {
  /// OPTIMIZATION STOPPING CRITERIA

  /** Stopping tolerance for the norm of the Riemannian gradient */
  double grad_norm_tol = 1e-2;

  /** Stopping tolerance for the norm of the preconditioned Riemannian
   * gradient */
  double preconditioned_grad_norm_tol = 1e-3;

  /** Stopping criterion based upon the relative decrease in function value */
  double rel_func_decrease_tol = 1e-5;

  /** Stopping criterion based upon the norm of an accepted update step */
  double stepsize_tol = 1e-3;

  /** Maximum permitted number of (outer) iterations of the Riemannian
   * trust-region method*/
  unsigned int max_iterations = 500;

  /** Maximum number of inner (truncated conjugate-gradient) iterations to
   * perform per out iteration */
  unsigned int max_tCG_iterations = 2000;

  /** Maximum elapsed computation time (in seconds) */
  double max_computation_time = std::numeric_limits<double>::max();

  /// FUSES PARAMETERS

  /** The initial level of the Riemannian Staircase
   * will be set to nrClasses+1 if default initialized */
  unsigned int r0 = 0;

  /** The maximum level of the Riemannian Staircase to explore */
  unsigned int rmax = 40;

  /** The maximum number of Lanczos iterations to admit for eigenvalue
   * computations */
  unsigned int max_eig_iterations = 10000;

  /** A numerical tolerance for acceptance of the minimum eigenvalue of Q -
   * Lambda(Y*) as numerically nonnegative; this should be a small magnitude
   * value e.g. 10^-4 */
  double min_eig_num_tol = 1e-4;

  /** The number of working vectors to use in the minimum eigenvalue
   * computation (using the implicitly-restarted Arnoldi algorithm); must be in
   * the range [1, (#nodes) * (#problem dimension) - 1] */
  unsigned int num_Lanczos_vectors = 20;

  /** The preconditioning strategy to use in the Riemannian trust-region
   * algorithm */
  Preconditioner precon = Jacobi;

  /** The rounding strategy to use in round_solution function  */
  Rounding round = WinnerTakeAll;

  /** Whether to print output as the algorithm runs */
  bool verbose = false;

  /** If this value is true, the Fuses algorithm will log and return the
   * entire sequence of iterates generated by the Riemannian Staircase */
  bool log_iterates = false;

  /** The number of threads to use for parallelization (assuming that Fuses
   * is built using a compiler that supports OpenMP */
  unsigned int num_threads = 1;

  /** Use unary factors for initialization (only used if Y is not provided to
   * FUSES) */
  InitializationType initializationType = FromUnary;

  /** If iterates are stored and this option is enabled, Fuses returns
   * best rounded solution rather than final rounded solution.*/
  bool use_best_rounded = false;
};

/* ================== FUSES: timing struct =========================== */
/** This struct contains the output of the FUSES algorithm */
struct FUSESTiming {
  /** The total elapsed computation time for the Fuses algorithm */
  double total;

  /** The elapsed computation time in initialization for FUSES */
  double initialization;

  /** The elapsed computation time in setting function optimization handles for
   * FUSES */
  double settingTNTHandles;

  /** A vector containing the sequence of elapsed times in the optimization at
   * each level of the Riemannian Staircase at which the corresponding function
   * values and gradients were obtained */
  std::vector<std::vector<double>> elapsed_optimization;

  /** A vector containing the elapsed time of the minimum eigenvalue computation
   * at each level of the Riemannian Staircase */
  std::vector<double> minimum_eigenvalue_computation;

  /** The elapsed computation time spent on rounding*/
  double rounding;

  void print() const {
    std::cout << "total_computation_time: " << total << "\n"
        << "initialization_time: " << initialization << "\n";
    UtilsGM::PrintVector<double>(elapsed_optimization.back(),
        "elapsed_optimization times (last r)");
    UtilsGM::PrintVector<double>(minimum_eigenvalue_computation,
        "minimum_eigenvalue_computation times (last r)");
    std::cout << "rounding time: " << rounding << "\n";
    std::cout << "settingTNTHandles time: " << settingTNTHandles << "\n";
  }
};

/* ================== FUSES: results struct =========================== */
/** This struct contains the output of the FUSES algorithm */
struct FUSESResult {
  /** The optimal value of the SDP relaxation */
  double SDPval;

  /** A global minimizer Yopt of the rank-restricted semidefinite relaxation */
  gtsam::Matrix Yopt;

  /** The norm of the Riemannian gradient at Yopt */
  double gradnorm;

  /** The minimum eigenvalue of the matrix S - Lambda(Yopt) */
  double lambda_min;

  /** The corresponding eigenvector of the minimum eigenvalue */
  gtsam::Vector v_min;

  /** The value of the objective function evaluated at rounded solution xhat */
  double Fxhat;

  /** The rounded solution xhat */
  gtsam::Matrix xhat;

  /** A vector containing the sequence of function values obtained during the
   * optimization at each level of the Riemannian Staircase */
  std::vector<std::vector<double>> function_values;

  /** A vector containing the sequence of norms of the Riemannian gradients
   * obtained during the optimization at each level of the Staircase*/
  std::vector<std::vector<double>> gradient_norms;

  /** A vector containing the sequence of minimum eigenvalues of the
   * certificate matrix constructed from the critical point recovered from the
   * optimization at each level of the Riemannian Staircase */
  std::vector<double> minimum_eigenvalues;

  /** Keep track of CPU time required by the different components in FUSES */
  FUSESTiming timing;

  /** If Fuses was run with log_iterates = true, this vector will contain the
   * entire sequence of iterates generated by the Riemannian Staircase */
  std::vector<std::vector<gtsam::Matrix>> iterates;

  FUSESStatus status = RS_ITER_LIMIT;
};

class FusesBase {
protected:
  /** A struct storing conditional random field */
  const CRFsegmentation::CRF& crf_;

  /** A struct storing variable Y in current iterate in Riemannian Staircase */
  gtsam::Matrix Y_;

  /** A cache variable to store the *Euclidean* gradient of Y at the current
   * iterate Y */
  gtsam::Matrix NablaF_Y_;

  /** A data matrix to store unary and binary terms */
  SpMatrix Q_;

  /** A struct storing optimization parameters */
  FUSESOpts options_;

  /** A struct storing optimization result */
  FUSESResult result_;

  /** Diagonal Jacobi preconditioner */
  DiagMatrix JacobiPrecon_;

  /** Vector mapping reduced crf class to the original class */
  std::vector<size_t> classRemap_;

  /** Vector mapping reduced crf class to the original class */
  std::vector<size_t> classConversion_;

  /** Set classRemap_ and classConversion_ vector so that FUSES can exclude
   * unused classes in CRF
   * (unused = no unary term is associated with that class) */
  void setReducedClassConversion() {
    // make sure this function will only run once
    if(classRemap_.size() > 0) {
      throw std::runtime_error("Reduced class conversion can only be set once");
    } else {
      // loop through unary factors to fill in present class
      std::vector<bool> presentClass(crf_.nrClasses, false);
      for(auto& unaryFactor : crf_.unaryFactors) {
        presentClass[unaryFactor.label] = true;
      }

      // fill in mapping vector (from original class to reduced class)
      // as well as class_conversion_ vector
      classRemap_.resize(crf_.nrClasses);
      size_t classIndex = 0;
      classConversion_.clear();
      for(size_t i=0; i<crf_.nrClasses; ++i) {
        if(presentClass[i]) {
          classRemap_[i] = classIndex;
          classConversion_.push_back(i);
          classIndex++;
        }
      }
    }
  }

  /** Retrieve original class from reduced class */
  void recoverLabels(std::vector<size_t>& labels) const {
    if(classConversion_.size() < 1) {
      return;
    } else {
      for(auto& label : labels) {
        label = classConversion_[label];
      }
    }
  }

public:
  /** Default constructor for serialization */
  FusesBase(const CRFsegmentation::CRF& crf) : crf_(crf) {}

  /** Accessory functions */
  const CRFsegmentation::CRF& getCRF() const {return crf_;}
  const gtsam::Matrix& getY() const {return Y_;}
  const SpMatrix& getDataMatrix() const {return Q_;}
  const FUSESResult& getResult() const {return result_;}
  const FUSESOpts& getOptions() const {return options_;}

  void setY(gtsam::Matrix& Y) {Y_ = Y;}

  /** Initialize the preconditioner matrix*/
  virtual void initializePrecon(){
    Eigen::VectorXd diag(Q_.rows());
    JacobiPrecon_ = diag.cwiseInverse().asDiagonal();
  }

  /** Given a matrix Y, this function computes and returns F(Y), the value of
   * the objective evaluated at Y */
  virtual double evaluate_objective(const gtsam::Matrix &Y) const {
    return (Y.transpose() * (Q_.selfadjointView<Eigen::Upper>() * Y)).trace();
  }

  /** Given a matrix Y, this function computes and returns nabla F(Y), the
   * *Euclidean* gradient of F at Y. */
  gtsam::Matrix Euclidean_gradient(const gtsam::Matrix &Y) const {
    return Q_.selfadjointView<Eigen::Upper>() * (2*Y);
  }
  /** Given a tangent matrix dotY in T_D(Y), the tangent space of the domain of
   * the optimization problem at Y, this function computes and returns the
   * *Euclidean* Hessian on dotY. */
  gtsam::Matrix Euclidean_Hessian_vector_product(const gtsam::Matrix &dotY)
      const {
    return Q_.selfadjointView<Eigen::Upper>() * (2*dotY);
  }

  /** Given a point Y in the domain D of the rank-r relaxation of the FUSES
   * optimization problem, this function computes and returns a matrix X in
   * R^(n x k) comprised of the labels of each node by rounding the upper right
   * block of Z = Y*Y'
   * note: this is a pure virtual function */
  virtual gtsam::Matrix round_solution(const gtsam::Matrix& Y) const = 0;

  /** Given a nrNodes x nrClasses matrix (with real entries), produces
   * an nrNodes x nrClasses binary matrix */
  gtsam::Matrix round_NxK_matrix(const gtsam::Matrix& M) const{
    switch(options_.round){
    case WinnerTakeAll :
      return round_solution_WTA(M); break;
    case RandomizedProj :
      return round_solution_RP(M); break;
    case Sampling :
      return round_solution_S(M); break;
    default:
      throw std::runtime_error("unrecognized rounding option (see Rounding types"
          " in Fuses2.h)");
      break;
    }
    return gtsam::Matrix(0,0); // to avoid warning - we never enter this case
  }

  /* ---------------------------------------------------------------------- */
  /** Helper function: used in round_solution
   * This function takes in a matrix storing doubles and apply row wise
   * *Winner-Take-All* strategy to round it to binary matrix of the same
   * dimmension */
  gtsam::Matrix round_solution_WTA(const gtsam::Matrix& P) const{
    int k = P.cols();
    int n = P.rows();
    // find the location of the maximum number in each row and set corresponding
    // entry in P to 1
    gtsam::Matrix::Index maxIndex;
    gtsam::Matrix X = gtsam::Matrix::Zero(n, k);

    for(int i=0; i<n; ++i) {
      P.row(i).maxCoeff(&maxIndex);
      X(i, maxIndex) = 1;
    }
    return X;
  }

  /* ---------------------------------------------------------------------- */
  /** Helper function: used in round_solution
   * Random projection rounding (experimental) */
  gtsam::Matrix round_solution_RP(const gtsam::Matrix& Z_ur) const{
    int k = Z_ur.cols();
    int n = Z_ur.rows();

    std::default_random_engine generator(1);
    std::normal_distribution<double> distribution(0.0, 1.0);

    gtsam::Matrix Xbest, Xtemp;
    gtsam::Matrix P(k, k);
    double fvalLowest = 0;
    for(int l=0; l<20; ++l) {
      for(int i=0; i<k; ++i) {
        for(int j=0; j<k; ++j) {
          P(i, j) = distribution(generator);
        }
      }
      Xtemp = round_solution_WTA(Z_ur * P);
      gtsam::Matrix Ytemp(n + k, k);
      Ytemp.block(0, 0, n, k) = Xtemp;
      Ytemp.block(n, 0, k, k) = gtsam::Matrix::Identity(k, k);
      double fval = evaluate_objective(Ytemp);
      if(fval < fvalLowest) {
        Xbest = Xtemp;
        fvalLowest = fval;
      }
    }
    return Xbest;
  }

  /* ---------------------------------------------------------------------- */
  /** Helper function: used in round_solution
   * This function takes in a matrix storing doubles and normalizes
   * the row to be label probabilities. Then samples potential solutions
   * and returns the one with the smallest cost */
  gtsam::Matrix round_solution_S(const gtsam::Matrix& P,
      const size_t nrSamples = 1000) const{
    int n = P.rows();
    int k = P.cols();

    // create distributions and random number generator
    std::default_random_engine generator;
    std::vector<std::discrete_distribution<int>> distributions =
        GetProbabilitiesFromMatrix(P);

    // compute WTA solution as baseline:
    gtsam::Matrix X_WTA = round_solution_WTA(P);
    std::vector<size_t> bestLabels = UtilsGM::GetLabelsFromMatrix(X_WTA, n, k);
    double bestRoundedCost = crf_.evaluate(bestLabels);
    // std::cout << "bestRoundedCost: " << bestRoundedCost << std::endl;

    // randomly sample potential solutions and evaluate cost
    std::vector<size_t> labels; labels.resize(n); // n labels
    for(size_t sample = 0; sample < nrSamples; sample++){
      for(size_t i=0; i<n; i++){ // for each node
        int l = distributions.at(i)(generator);
        if((l<0) || (l > k-1)){ throw std::runtime_error("round_solution_S: "
            "Label i outside range");}
        labels.at(i) = l;
      }
      double cost = crf_.evaluate(labels);
      // std::cout << "cost: " << cost << std::endl;
      if(cost < bestRoundedCost){
        std::cout << "sampling improved cost from " << bestRoundedCost << " to "
            << cost << std::endl;
        bestRoundedCost = cost;
        bestLabels = labels;
      }
    }
    // return in matrix form
    return UtilsGM::GetMatrixFromLabels(bestLabels, n, k);
  }

  /** extracts probabilities from a matrix, but setting to zero the negative
   * entries */
  static std::vector<std::discrete_distribution<int>>
      GetProbabilitiesFromMatrix(const gtsam::Matrix P) {
    int n = P.rows();
    int k = P.cols();

    std::vector<std::discrete_distribution<int>> distributions;
    distributions.reserve(n);

    // populate probabilities
    std::vector<double> prob; prob.resize(k);
    // normalize rows to be probabilities:
    for(size_t r=0; r<n; r++){
      double sumRow = 0.0;
      for(size_t c = 0; c<k; c++){
        if(P(r,c) < 0){
          prob.at(c) = 0.0; // set negative entries to zero probability
        }else{
          sumRow += P(r,c);
          prob.at(c) = P(r,c); // does not need to be normalized
        }
      }
      if(sumRow == 0.0){
        throw std::runtime_error("round_solution_S: matrix has row sum equal to"
            " zero"); break;
      }
      std::discrete_distribution<int> distributions_r(prob.begin(), prob.end());
      distributions.push_back(distributions_r);
    }
    return distributions;
  }


  /* ---------------------------------------------------------------------- */
  /** Use iterated to compute intermediate rounded costs a posteriori.
   */
  std::vector<double> computeIntermediateRoundedObjectives() const {
    std::vector<double> fvalRounded;
    if(options_.log_iterates) { // log value after rounding for each iteration
      for(size_t i=0; i<result_.iterates.size(); i++) {
        for(auto j=result_.iterates[i].begin(); j!=result_.iterates[i].end();
            ++j) {
          // Evaluate objective function at ROUNDED solution
          fvalRounded.push_back(crf_.evaluate(round_solution(*j)));
        }
        // TODO LC: check this
        if(fvalRounded.size() != result_.function_values.back().size()) {
          fvalRounded.push_back(result_.Fxhat);
        }
      }
    }else{
      // if we cannot compute a rounded value since we do not have the iterates,
      // we put zero
      for(size_t i=0; i<result_.function_values.back().size(); i++) {
        fvalRounded.push_back(0.0); // padding
      }
    }
    return fvalRounded;
  }

  /* ---------------------------------------------------------------------- */
  /** Helper function: to convert a result matrix to a std vector of integers
   *  Note that this code assumes that resultMatrix is already rounded.
   */
  std::vector<size_t> convertMatrixToVec(const gtsam::Matrix& resultMatrix)
      const {
    std::vector<size_t> labels = UtilsGM::GetLabelsFromMatrix(resultMatrix,
        crf_.nrNodes, crf_.nrClasses);
    recoverLabels(labels);
    return labels;
  }

  /** Given a csv file storing two sets of xhat1 and xhat2, this function
   * compares result_.xhat with these values. The function returns true if
   *    (1) the difference between F(xhat) and either F(xhat1) or F(xhat2) is
   *    within -errorRate * F(xhat) (the minus sign is used because F(xhat) is
   *    negative)
   *    (2) the difference between F(Yopt) and either fval1 or fval2 is within
   *    -errorRate * F(Yopt). These are the objective function values before
   *    rounding
   *    (3) The infinity norm, norm(Yopt*Yopt'-Z) < errorRate, where Z is the
   *    global minimizer of method2
   * This needs to be used in conjunction with writeDataToCSV.m to ensure
   * correct input data format.
   */
  bool compare(const std::string& filename, double errorRate = 0.001) const {
    // Load file
    std::ifstream file(filename);
    std::string line;
    std::vector<std::string> fields;
    if(!file.good()) {
      std::cout <<  "Could not read file " << filename << std::endl;
      throw std::runtime_error("File not found.");
    }

    // Load fval and get fuses_fval
    std::getline(file, line, ',');
    double fval1 = std::stod(line); // exact objective func value (cplex)
    std::getline(file, line);
    double fval2 = std::stod(line); // reference objective func value before
                                    // rounding
    double fuses_fval = evaluate_objective(result_.Yopt);

    // Get parameters from crf_ to initialize xhat matrices
    int k = crf_.nrClasses;
    int n = crf_.nrNodes;
    gtsam::Matrix xhat1 = gtsam::Matrix(n, k);
    gtsam::Matrix xhat2 = gtsam::Matrix(n, k);

    // Read xhat matrices from the input file
    for(int i=0; i<n; ++i) {
      std::getline(file, line);
      boost::split(fields, line, boost::is_any_of(","));

      for(int j=0; j<2*k; ++j) {
        if(j < k) {
          xhat1(i, j) = std::stoi(fields[j]);
        } else {
          xhat2(i, j-k) = std::stoi(fields[j]);
        }
      }
    }

    // Read Z matrix from the input file
    // Ideally, Z = Yopt*Yopt' so that fuses agrees with method 2
    gtsam::Matrix Z = gtsam::Matrix(n+k, n+k);
    for(int i=0; i<n+k; ++i) {
      getline(file, line);
      boost::split(fields, line, boost::is_any_of(","));

      for(int j=0; j<n+k; ++j) {
        Z(i, j) = std::stod(fields[j]);
      }
    }
    file.close();

    // Compare costs and labels
    gtsam::Matrix Xhat1(n + k, k);
    Xhat1.block(0, 0, n, k) = xhat1;
    Xhat1.block(n, 0, k, k) = gtsam::Matrix::Identity(k, k);
    double Fxhat1 = evaluate_objective(Xhat1);
    std::cout << "Method 1:" << std::endl;
    std::cout << "  Optimal value (before rounding) = " << fval1
        << "  Gap between FUSES and method 1 = "
        << fuses_fval - fval1 << std::endl;
    std::cout << "  Optimal value (after rounding) = " << Fxhat1
        << "  Gap between FUSES and method 1 = "
        << result_.Fxhat - Fxhat1 << std::endl;
    std::cout << "  Label difference between FUSES and method 1: "
        << (result_.xhat - xhat1).lpNorm<1>()/2 << std::endl;

    gtsam::Matrix Xhat2(n + k, k);
    Xhat2.block(0, 0, n, k) = xhat2;
    Xhat2.block(n, 0, k, k) = gtsam::Matrix::Identity(k, k);
    double Fxhat2 = evaluate_objective(Xhat2);
    double infNorm =
        (result_.Yopt * result_.Yopt.transpose() - Z).cwiseAbs().maxCoeff();
    std::cout << "Method 2:" << std::endl;
    std::cout << "  Optimal value (before rounding) = " << fval2
        << "  Gap between FUSES and method 2 = "
        << fuses_fval - fval2 << std::endl;
    std::cout << "  Optimal value (after rounding) = " << Fxhat2
        << "  Gap between FUSES and method 2 = "
        << result_.Fxhat - Fxhat2 << std::endl;
    std::cout << "  Label difference between FUSES and method 2: "
        << (result_.xhat - xhat2).lpNorm<1>()/2 << std::endl;
    std::cout << "  Infinity norm of Yopt*Yopt'-Z: " << infNorm << std::endl
        << std::endl;

    // compare objective function before rounding
    if(abs(fuses_fval - fval1) > -errorRate * fuses_fval
        && abs(fuses_fval - fval2) > -errorRate * fuses_fval) {
      return false;
      // compare objective function after rounding
    } else if(abs(result_.Fxhat - Fxhat1) > -errorRate * result_.Fxhat
        && abs(result_.Fxhat - Fxhat2) > -errorRate * result_.Fxhat) {
      return false;
    } else {
      return true;
    }
  }
};

} // namespace FUSES
