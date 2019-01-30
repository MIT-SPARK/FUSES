#include "FUSES/StiefelMixedProduct.h"
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd StiefelMixedProduct::project(const Eigen::MatrixXd &A) const {

  // We use a generalization of the well-known SVD-based projection for the
  // orthogonal and special orthogonal groups; see for example Proposition 7
  // in the paper "Projection-Like Retractions on Matrix Manifolds" by Absil
  // and Malick.
  Eigen::MatrixXd P(n_ + k_, r_);

//  // For 1-D block, normalize is faster than SVD
//#pragma omp parallel for
//  for (unsigned int i = 0; i < n_; ++i) {
//    P.row(i) = A.row(i).normalized();
//  }
  P.topRows(n_) = A.topRows(n_).rowwise().normalized(); // normalize first n rows to have norm 1

  // Compute the (thin) SVD of the last block of A
  Eigen::JacobiSVD<Eigen::MatrixXd> SVD(
      A.block(n_, 0, k_, r_), Eigen::ComputeThinU | Eigen::ComputeThinV);

  // Set the last block of P to the SVD-based projection of the last block of A
  P.block(n_, 0, k_, r_) = SVD.matrixU() * SVD.matrixV().transpose();

  return P;
}
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd
StiefelMixedProduct::SymBlockDiagProduct(const Eigen::MatrixXd &A,
                                    const Eigen::MatrixXd &B,
                                    const Eigen::MatrixXd &C) const {

  Eigen::SparseMatrix<double> symBlockDiag(n_ + k_, n_ + k_);
  symBlockDiag.reserve(n_ + k_ * k_);

  // Insert the first n diagonal entries
  for (int i=0; i<n_; ++i) {
    symBlockDiag.insert(i, i) = A.row(i).dot( B.row(i) );
  }

  // Insert the last k x k diagonal block
  // (since this block is symmetric, only the lower part is needed)
  Eigen::MatrixXd P = A.bottomRows(k_) * B.bottomRows(k_).transpose();
  Eigen::MatrixXd temp = P + P.transpose();
  for (int i=0; i<k_; ++i) {
    for (int j=0; j<i+1; ++j) {
      symBlockDiag.insert(n_ + i, n_ + j) = .5 * temp(i, j);
    }
  }
  symBlockDiag.makeCompressed();

  return symBlockDiag.selfadjointView<Eigen::Lower>() * C;
}
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd StiefelMixedProduct::Proj(const Eigen::MatrixXd &Y,
    const Eigen::MatrixXd &V) const {
  return V - SymBlockDiagProduct(V, Y, Y);
}

Eigen::MatrixXd StiefelMixedProduct::retract(const Eigen::MatrixXd &Y,
                                        const Eigen::MatrixXd &V) const {
  return project(Y + V);
}
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd StiefelMixedProduct::random_sample() const {
  // Generate a matrix of the appropriate dimension by sampling its elements
  // from the standard Gaussian
  std::default_random_engine generator;
  std::normal_distribution<double> g;

  Eigen::MatrixXd R(n_ + k_, r_);
  for (unsigned int r = 0; r < n_ + k_; r++)
    for (unsigned int c = 0; c < r_; c++)
      R(r, c) = g(generator);
  return project(R);
}
