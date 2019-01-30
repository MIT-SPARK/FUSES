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

#include "FUSES/FusesBase.h"
#include "FUSES/StiefelMixedProduct.h"

namespace FUSES {

/* ============================= FUSES =========================== */
class Fuses : public FusesBase {
 public:
  /** Fuses constructor: taking crf and options (optional)
   * Will generate a random initial guess */
  Fuses(const CRFsegmentation::CRF& crf, const boost::optional<FUSESOpts&>
    options = boost::none);

  /** Fuses constructor: taking initial guess in matrix form (only for expert
   * users, will set r0 according to Y0) and options (optional) */
  Fuses(const CRFsegmentation::CRF& crf, const gtsam::Matrix &Y0,
      const boost::optional<FUSESOpts&> options = boost::none);

  /** Fuses constructor: taking initial guess in vector form nodeLabels[i]
   * corresponds to class of node i and options (optional) */
  Fuses(const CRFsegmentation::CRF& crf, const std::vector<size_t>& nodeLabels,
      const boost::optional<FUSESOpts&> options = boost::none);

  // TODO: check destructor
  //  /** Fuses destructor */
  //  ~Fuses();

  /** The underlying manifold in which the multi-class labels lie in the
   * rank-restricted Riemannian optimization problem.*/
  StiefelMixedProduct shp_;

  // TODO: check IncompleteCholesky later
  //  /** Incomplete Cholesky Preconditioner */
  //  IncompleteCholeskyFactorization *iChol_ = nullptr;

  /** Generate random initial guess for Y */
  void Y_rand();

  /** Generate initial guess using unary factors in CRF + assign label randomly
   * if no unary factor is provided */
  void Y_from_unary();

  /** Initialize data member Y_ with a vector */
  void setYfromVector(const std::vector<size_t>& nodeLabels);

  /** Initialize SteifelMixedProduct manifold dimensions */
  void initializeSMP();

  /** Get data matrix Q from CRF */
  void initializeQandPrecon(bool exactDiag = false);

  /** Set the maximum rank of the rank-restricted semidefinite relaxation */
  void set_relaxation_rank(unsigned int rank);

  /** Returns the relaxation rank r of this problem */
  unsigned int get_relaxation_rank() const;

  /** Given a matrix Y in the domain D of the FUSES optimization problem and
   * the *Euclidean* gradient nablaF_Y at Y, this function computes and
   * returns the *Riemannian* gradient grad F(Y) of F at Y. If nablaF_Y is not
   * provided, it will be computed using Euclidean_gradient function. */
  gtsam::Matrix Riemannian_gradient(const gtsam::Matrix &Y,
      const boost::optional<gtsam::Matrix&> nablaF_Y = boost::none) const;

  /** Given a matrix Y in the domain D of the FUSES optimization problem, the
   * *Euclidean* gradient nablaF_Y of F at Y, and a tangent matrix dotY in
   * T_D(Y), the tangent space of the domain of the optimization problem at Y,
   * this function computes and returns Hess F(Y)[dotY], the action of the
   * Riemannian Hessian on dotY */
  gtsam::Matrix Riemannian_Hessian_vector_product(const gtsam::Matrix& Y,
      const gtsam::Matrix& nablaF_Y, const gtsam::Matrix& dotY) const;

  /** Given a matrix Y in the domain D of the FUSES optimization problem, and
   * a tangent matrix dotY in T_D(Y), the tangent space of the domain of the
   * optimization problem at Y, this function computes and returns Hess
   * F(Y)[dotY], the action of the Riemannian Hessian on dotY */
  gtsam::Matrix Riemannian_Hessian_vector_product(const gtsam::Matrix& Y,
      const gtsam::Matrix& dotY) const;

  /** Given a matrix Y in the domain D of the FUSES optimization problem and a
   * tangent vector dotY in T_Y(E), the tangent space of Y considered as a
   * generic matrix, this function computes and returns the orthogonal
   * projection of dotY onto T_D(Y), the tangent space of the domain D at Y */
  gtsam::Matrix tangent_space_projection(const gtsam::Matrix &Y,
      const gtsam::Matrix &dotY) const;

  /** Given a matrix Y in the domain D of the FUSES optimization problem and a
   * tangent vector dotY in T_D(Y), this function returns the point Yplus in D
   * obtained by retracting along dotY */
  gtsam::Matrix retract(const gtsam::Matrix& Y,
      const gtsam::Matrix& dotY) const;

  // TODO: testing here
  gtsam::Matrix precondition(const gtsam::Matrix& Y,
      const gtsam::Matrix& dotY) const;

  /** Given a point Y in the domain D of the rank-r relaxation of the FUSES
   * optimization problem, this function computes and returns a matrix X in
   * R^(n x k) comprised of the labels of each node by rounding the upper right
   * block of Z = Y*Y' */
  gtsam::Matrix round_solution(const gtsam::Matrix& Y) const;

  /** Given a critical point Y of the rank-r relaxation of the FUSES
   * optimization problem, this function computes and returns a (n+k) x (n+k)
   * matrix using Q - Lambda. Lambda = SymBlockDiag(Q*Y*Y') is a block-diagonal
   * Lagrange multiplier matrix associated with the orthonormality
   * constraints */
  SpMatrix compute_Q_minus_Lambda(const gtsam::Matrix &Y) const;

  /** Given a critical point Y in the domain of the optimization problem, this
   * function computes the smallest eigenvalue lambda_min of S - Lambda and its
   * associated eigenvector v.  Returns a Boolean value indicating whether the
   * Lanczos method used to estimate the smallest eigenpair converged to
   * within the required tolerance. */
  bool compute_Q_minus_Lambda_min_eig(const gtsam::Matrix& Y,
      double& min_eigenvalue, gtsam::Vector& min_eigenvector,
      unsigned int max_iterations = 10000,
      double min_eigenvalue_nonnegativity_tolerance = 1e-5,
      unsigned int num_Lanczos_vectors = 20) const;

  /** Helper function: used in the Riemannian Staircase to escape from a saddle
   * point.  Here:
   *
   * - problem is the specific special Euclidean synchronization problem we are
   *     attempting to solve
   * - Y is the critical point (saddle point) obtained at the current level of
   *     the Riemannian Staircase
   * - lambda_min is the (negative) minimum eigenvalue of the matrix Q - Lambda
   * - v_min is the eigenvector corresponding to lambda_min
   * - gradient_tolerance is a *lower bound* on the norm of the Riemannian
   * gradient grad F(Yplus)
   *     in order to accept a candidate point Xplus as a valid solution
   *
   * Post-condition:  This function returns a Boolean value indicating whether
   * it was able to successfully escape from the saddle point, meaning it found
   * a point Yplus satisfying the following two criteria:
   *
   *  (1)  F(Yplus) < F(Y), and
   *  (2)  || grad F(Yplus) || > gradient_tolerance
   *
   * Condition (2) above is necessary to ensure that the optimization
   * initialized at the next level of the Riemannian Staircase does not
   * immediately terminate due to the gradient stopping tolerance being
   * satisfied.
   *
   * Precondition: the relaxation rank r must be 1 greater than the number of
   * rows of Y (i.e., the relaxation rank of must already be set for the *next*
   * level of the Riemannian Staircase when this function is called.
   *
   * Postcondition: If this function returns true, then upon termination Yplus
   * contains the point at which to initialize the optimization at the next
   * level of the Riemannian Staircase
   */
  bool escape_saddle(const gtsam::Matrix& Y, double lambda_min,
      const gtsam::Vector& v_min, double gradient_tolerance,
      double preconditioned_gradient_tolerance, gtsam::Matrix* Yplus);

  /** This function solves the optimization problem on Riemannian manifolds as
   * specified by options_ data member and write the solution as well as the
   * solution status into result_.
   */
  void solve();
}; // Class Fuses

} // namespace FUSES
