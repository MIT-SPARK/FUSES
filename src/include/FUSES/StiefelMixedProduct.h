/** This lightweight class models the geometry of M = St(1, r)^n X St(k, r),
 * the product manifold of n (transposed) Stiefel manifold St(1, r) and a
 * (transposed) Stiefel manifold St(K, r). Elements of this manifold (and its
 * tangent spaces) are represented as (n+k) x r matrices of type 'MatrixType'.
 *
 * Copyright (C) 2018 by Siyi Hu
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <random> // For sampling random points on the manifold

class StiefelMixedProduct {

private:
  // Dimension of ambient Euclidean space containing the frames
  unsigned int r_;

  // Number of copies of St(1, r) in the product
  unsigned int n_;

  // Number of vectors in the last orthonormal k-frame
  unsigned int k_;

public:
  /// CONSTRUCTORS AND MUTATORS

  // Default constructor -- sets all dimensions to 0
  StiefelMixedProduct() {}

  StiefelMixedProduct(unsigned int r, unsigned int n, unsigned int k)
      : r_(r), n_(n), k_(k) {}

  void set_r(unsigned int r) { r_ = r; }
  void set_n(unsigned int n) { n_ = n; }
  void set_k(unsigned int k) { k_ = k; }

  /// ACCESSORS
  unsigned int get_r() const { return r_; }
  unsigned int get_n() const { return n_; }
  unsigned int get_k() const { return k_; }

  /// GEOMETRY
  /** Given a generic matrix A in R^{(n+k) x r}, this function computes the
   * projection of A onto M (closest point in the Frobenius norm sense).  */
  Eigen::MatrixXd project(const Eigen::MatrixXd &A) const;

  /** Helper function -- this computes and returns the product
   *  R = SymBlockDiag(A * B^T) * C
   * where A, B, and C are (n+k) x r matrices.
   */
  Eigen::MatrixXd SymBlockDiagProduct(const Eigen::MatrixXd &A,
                                      const Eigen::MatrixXd &B,
                                      const Eigen::MatrixXd &C) const;

  /** Given an element Y in M and a matrix V in R^{(n+k) x r}, this function
   * computes and returns the projection of V onto T_Y(M), the tangent space
   * of M at Y.*/
  Eigen::MatrixXd Proj(const Eigen::MatrixXd &Y, const Eigen::MatrixXd &V) const;

  /** Given an element Y in M and a tangent vector V in T_Y(M), this function
   * computes the retraction along V at Y using the QR-based retraction
   * specified in eq. (4.8) of Absil et al.'s  "Optimization Algorithms on
   * Matrix Manifolds").
   */
  Eigen::MatrixXd retract(const Eigen::MatrixXd &Y,
                          const Eigen::MatrixXd &V) const;

  /** Sample a random point on M */
  Eigen::MatrixXd random_sample() const;
};
