#include "FUSES/StiefelProduct.h"
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd StiefelProduct::project(const Eigen::MatrixXd &A) const {

  Eigen::MatrixXd P(n_, r_);
  P = A.rowwise().normalized(); // normalize first n rows to have norm 1
  return P;
}
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd // typically computed for Y, nablaF_Y, dotY
StiefelProduct::SymBlockDiagProduct(const Eigen::MatrixXd &A,
                                    const Eigen::MatrixXd &B,
                                    const Eigen::MatrixXd &C) const {

  // // Full but potentially slower math
  //  Eigen::SparseMatrix<double> symBlockDiag(n_, n_);
  //  symBlockDiag.reserve(n_);
  //  // Insert the n diagonal entries
  //  for (int i=0; i<n_; ++i) { // LC: to check
  //    symBlockDiag.insert(i, i) = A.row(i).dot( B.row(i) );
  //  }
  //  symBlockDiag.makeCompressed();
  //  return symBlockDiag.selfadjointView<Eigen::Lower>() * C;

  Eigen::MatrixXd S(C.rows(),C.cols());
  for (int i=0; i<n_; ++i) { // LC: to check
    S.row(i) = ( A.row(i).dot( B.row(i) ) )*C.row(i);
  }
  return S;
}
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd StiefelProduct::Proj(const Eigen::MatrixXd &Y,
    const Eigen::MatrixXd &V) const {
  return V - SymBlockDiagProduct(V, Y, Y);
}
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd StiefelProduct::retract(const Eigen::MatrixXd &Y,
                                        const Eigen::MatrixXd &V) const {
  return project(Y + V);
}
/* ---------------------------------------------------------------------- */
Eigen::MatrixXd StiefelProduct::random_sample() const {
  // Generate a matrix of the appropriate dimension by sampling its elements
  // from the standard Gaussian
  std::default_random_engine generator;
  std::normal_distribution<double> g;
  Eigen::MatrixXd R(n_, r_);
  for (unsigned int r = 0; r < n_; r++)
    for (unsigned int c = 0; c < r_; c++)
      R(r, c) = g(generator);
  return project(R);
}
