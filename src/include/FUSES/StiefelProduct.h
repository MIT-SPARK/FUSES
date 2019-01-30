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

class StiefelProduct {

private:
  // Dimension of ambient Euclidean space containing the frames
  unsigned int r_;

  // Number of copies of St(1, r) in the product
  unsigned int n_;

public:
  /// CONSTRUCTORS AND MUTATORS

  // Default constructor -- sets all dimensions to 0
  StiefelProduct() {}

  StiefelProduct(unsigned int r, unsigned int n)
      : r_(r), n_(n) {}

  void set_r(unsigned int r) { r_ = r; }
  void set_n(unsigned int n) { n_ = n; }

  /// ACCESSORS
  unsigned int get_r() const { return r_; }
  unsigned int get_n() const { return n_; }

  /// GEOMETRY
  Eigen::MatrixXd project(const Eigen::MatrixXd &A) const;

  Eigen::MatrixXd SymBlockDiagProduct(const Eigen::MatrixXd &A,
                                      const Eigen::MatrixXd &B,
                                      const Eigen::MatrixXd &C) const;

  Eigen::MatrixXd Proj(const Eigen::MatrixXd &Y, const Eigen::MatrixXd &V) const;

  Eigen::MatrixXd retract(const Eigen::MatrixXd &Y,
                          const Eigen::MatrixXd &V) const;

  Eigen::MatrixXd random_sample() const;
};
