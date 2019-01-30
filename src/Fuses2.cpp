#include "FUSES/Fuses2.h"

using namespace std;
using namespace gtsam;
using namespace Spectra;
using namespace CRFsegmentation;
using namespace myUtils;

namespace FUSES {
/* ---------------------------------------------------------------------- */
Fuses2::Fuses2(const CRF& crf, const boost::optional<FUSESOpts&> options)
  : FusesBase(crf) {
//  modifyCRF(); // reduce the size of CRF // TODO
  if (options) {
    options_ = *options;
    // use default r0 = 2 if not initialized
    if (options_.r0 < 1) {
      options_.r0 = 2;
    }
  } else {
    options_ = FUSESOpts();
    options_.r0 = 2;
  }
  auto myTime = Timer::tic();
  initializeSMP();
  switch(options_.initializationType){
  case FromUnary :
	  Y_from_unary(); break;
  case RandomClasses :
	  Y_rand(); break;
  case Random :
	  Y_ = sp_.random_sample(); break;
  default:
	  throw runtime_error("unrecognized initalization option (see initalization"
	      " types in FusesBase.h)"); break;
  }
  initializeQandPrecon();
  result_.timing.initialization = Timer::toc(myTime);
}
/* ---------------------------------------------------------------------- */
Fuses2::Fuses2(const CRF& crf, const Matrix &Y0,
    const boost::optional<FUSESOpts&> options) : FusesBase(crf)  {
  if(Y0.rows() != crf_.nrNodes*crf_.nrClasses+1) {
    cout <<  "Fuses2: Size of Y0 is incorrect = " << Y0.rows()
        << ", expecting " << crf_.nrNodes*crf_.nrClasses + 1 << endl;
    throw runtime_error("Wrong Y0 size.");
  }
  Y_ = Y0;
  if (options) {
    options_ = *options;
  } else {
    options_ = FUSESOpts();
  }
  options_.r0 = Y_.cols();
  auto myTime = Timer::tic();
  initializeSMP();
  initializeQandPrecon();
  result_.timing.initialization = Timer::toc(myTime);
}
/* ---------------------------------------------------------------------- */
Fuses2::Fuses2(const CRF& crf, const std::vector<size_t>& nodeLabels,
    const boost::optional<FUSESOpts&> options) : FusesBase(crf)  {
  if (options) {
    options_ = *options;
  } else {
    options_ = FUSESOpts();
    options_.r0 = 2;
  }
  setYfromVector(nodeLabels);
  auto myTime = Timer::tic();
  initializeSMP();
  initializeQandPrecon();
  result_.timing.initialization = Timer::toc(myTime);
}
/* ---------------------------------------------------------------------- */
// TODO: check IncompleteCholesky later
//Fuses2::~Fuses2() {
//  if (iChol_)
//    delete iChol_;
//}
/* ---------------------------------------------------------------------- */
void Fuses2::Y_from_unary() {
  // last element equal to 1
  Y_ = - Matrix::Ones(crf_.nrNodes * crf_.nrClasses + 1, options_.r0);
  //add unary factors to Y_
  std::vector<size_t> nodesWithUnary;
  for(auto unaryFactor : crf_.unaryFactors) {
    Y_(x_ind_K(unaryFactor.node,unaryFactor.label), 0) = +1;
    nodesWithUnary.push_back(unaryFactor.node);
  }
  for(int i=0; i<crf_.nrNodes; ++i) { // TODO LC: this loop is very inefficient
    auto it = find (nodesWithUnary.begin(), nodesWithUnary.end(), i);
    if (it == nodesWithUnary.end()){ // no unary given, we randomly assign label
      int nodeLabel = rand() % crf_.nrClasses;
      Y_(x_ind_K(i,nodeLabel), 0) = +1;
    }
  }
  Y_.rightCols(1) = Matrix::Zero(crf_.nrNodes * crf_.nrClasses+1, 1);
  Y_.bottomLeftCorner(1,1) = Matrix::Identity(1,1); // add last element = 1
}
/* ---------------------------------------------------------------------- */
void Fuses2::Y_rand() {
  // last element equal to 1
  Y_ = - Matrix::Ones(crf_.nrNodes * crf_.nrClasses + 1, options_.r0);
  // randomly assign label to other nodes
  for(int i=0; i<crf_.nrNodes; ++i) {
    int nodeLabel = rand() % crf_.nrClasses;
    Y_(x_ind_K(i,nodeLabel), 0) = +1;
  }
  // add identity block
  Y_.rightCols(1) = Matrix::Zero(crf_.nrNodes * crf_.nrClasses+1, 1);
  Y_.bottomLeftCorner(1,1) = Matrix::Identity(1,1); // add last element = 1
}
/* ---------------------------------------------------------------------- */
void Fuses2::setYfromVector(const std::vector<size_t>& nodeLabels) {
  if(nodeLabels.size() != crf_.nrNodes) {
    cout <<  "Size of nodeLabels = " << nodeLabels.size() << ", expecting "
         << crf_.nrNodes << endl;
    throw runtime_error("Wrong input size.");
  }

  // last element equal to 1
  Y_ = - Matrix::Ones(crf_.nrNodes * crf_.nrClasses + 1, options_.r0);
  // write nodeLabels in vector form
  for(int i=0; i<crf_.nrNodes; ++i) {
//     std::cout << " ( " << i << "," << nodeLabels[i] << ") -> x_ind_K: "
//         << x_ind_K(i, nodeLabels[i]) << std::endl;
    Y_(x_ind_K(i, nodeLabels[i]), 0) = +1;
  }
  // add identity block
  Y_.bottomLeftCorner(1,1) = Matrix::Identity(1,1); // add last element = 1
  // UtilsOpenCV::PrintVector<int>(nodeLabels,"nodeLabels");
}
/* ---------------------------------------------------------------------- */
void Fuses2::initializeSMP() {
  sp_ = StiefelProduct(options_.r0, crf_.nrNodes*crf_.nrClasses+1);
}
/* ---------------------------------------------------------------------- */
void Fuses2::initializeQandPrecon(bool exactDiag) {
  int n = crf_.nrNodes;
  int k = crf_.nrClasses;

  // Initialize Q and allocate room for nonzero elements
  // Only the upper half of Q is needed for all computations in Fuses2
  Q_.resize(n*k+1, n*k+1);
  Q_.reserve(k * (crf_.binaryFactors.size() + 2*n + k));

  // unary factors
  Eigen::VectorXd diag;
  if(exactDiag) {// initialize an n-vector to 0
    diag = Eigen::VectorXd::Constant(Q_.rows(), 0);
  } else {       // initialize an n-vector to 0.01
    diag = Eigen::VectorXd::Constant(Q_.rows(), 1);
  }
  //
  // Q = [H       1/2 g]
  //     [1/2g        O]   with H = nk x nk and g = nk x 1
  //
  // binary factors
  for(auto const& binaryFactor : crf_.binaryFactors){
    // CRF only stores half of the binary terms
    // (i.e. if edge(i,j) is stored, then edge(j,i) will not be stored but
    // should have the same weight)
    for(size_t j=0; j<k; j++){
      // instead of the kronecker product we populate manually
      double entry = -0.25*binaryFactor.weight;
      Q_.insert(binaryFactor.firstNode*k + j, binaryFactor.secondNode*k + j)
          = entry;
      Q_.insert(binaryFactor.secondNode*k + j, binaryFactor.firstNode*k + j)
          = entry;
      if(exactDiag) {
        diag[binaryFactor.firstNode*k+j] += fabs(entry); // (-entry);
        diag[binaryFactor.secondNode*k+j] += fabs(entry); // (-entry);
      }
    }
  }
  // unary factors
  Vector g = Vector::Zero(n*k,1); // temporary
  // populate entries in Q corresponding to unary factors
  for(auto const& unaryFactor : crf_.unaryFactors) {
    g(unaryFactor.node*k + unaryFactor.label) = -1.0 * unaryFactor.weight;
  }
  Vector g_tilde = (0.5*4.0)*(Q_.block(0, 0, n*k, n*k)*Vector::Ones(n*k, 1))
      + 0.5*g; // 4 compensates the 0.25 in the computation of Q.block
  // populate entries in Q corresponding to unary factors
  for(size_t i=0; i<g_tilde.size(); i++){
    double entry = 0.5 * g_tilde(i);
    Q_.insert(i, n*k) = entry; // last column
    diag[n*k] += fabs(entry); // (-entry);
  }
  Q_.makeCompressed();

  // constant offset in the cost that differentiates -1/+1 formulation form
  // the CRF cost evaluation
  Vector u = Vector::Ones(n*k);
  offset = (0.25*4.0)* // 4 compensates the 0.25 in the computation of Q.block
      ( u.transpose() *  Q_.block(0,0,n*k,n*k) * u ).trace() + 0.5 *u.dot(g);

  // Compute preconditioner matrix
  if(options_.precon == Jacobi) {
    JacobiPrecon_ = diag.cwiseInverse().asDiagonal();
  }
  //std::cout << "diag.cwiseInverse()" << diag.cwiseInverse()  <<std::endl;
}
/* ---------------------------------------------------------------------- */
void Fuses2::set_relaxation_rank(unsigned int rank) {sp_.set_r(rank);}
/* ---------------------------------------------------------------------- */
unsigned int Fuses2::get_relaxation_rank() const {return sp_.get_r();}
/* ---------------------------------------------------------------------- */
Matrix Fuses2::Riemannian_gradient(const Matrix &Y,
    const boost::optional<Matrix&> nablaF_Y) const {
  if (nablaF_Y) {
    return sp_.Proj(Y, *nablaF_Y);
  } else {
    return sp_.Proj(Y, Euclidean_gradient(Y));
  }
}
/* ---------------------------------------------------------------------- */
Matrix Fuses2::Riemannian_Hessian_vector_product(const Matrix &Y,
    const Matrix &nablaF_Y, const Matrix &dotY) const {
  return sp_.Proj(Y, Euclidean_Hessian_vector_product(dotY) -
      sp_.SymBlockDiagProduct(nablaF_Y, Y, dotY));
}
/* ----------------------------------------------------------------------*/
Matrix Fuses2::Riemannian_Hessian_vector_product(const Matrix &Y,
    const Matrix &dotY) const {
  return Riemannian_Hessian_vector_product(Y, Euclidean_gradient(Y), dotY);
}
/* ---------------------------------------------------------------------- */
Matrix Fuses2::tangent_space_projection(const Matrix &Y,
    const Matrix &dotY) const {
  return sp_.Proj(Y, dotY);
}
/* ---------------------------------------------------------------------- */
Matrix Fuses2::retract(const Matrix &Y, const Matrix &dotY) const {
  return sp_.retract(Y, dotY);
}
/* ----------------------------------------------------------------------*/
Matrix Fuses2::precondition(const Matrix &Y, const Matrix &dotY) const {
  if(options_.precon == None) {
    return dotY;
  } else if (options_.precon == Jacobi) {
    return tangent_space_projection(Y, JacobiPrecon_ * dotY);
  }
  // TODO: check IncompleteCholesky later
  //  else { // preconditioner = IncompleteCholesky
  //    return tangent_space_projection(Y, iChol_->solve(dotY));
  //  }
  return Matrix(0,0); // to avoid warning - we never enter this case
}
/* ---------------------------------------------------------------------- */
Matrix Fuses2::round_solution(const Matrix &Y) const {
  // compute the upper left block of matrix Z = Y*Y'
  Matrix z01 = 0.5 * (Y.topRows(crf_.nrNodes*crf_.nrClasses) *
      Y.bottomRows(1).transpose()
      + Matrix::Ones(crf_.nrNodes*crf_.nrClasses, 1));
  Matrix Z_ur = UtilsGM::Reshape(z01,crf_.nrClasses,crf_.nrNodes).transpose();
  return round_NxK_matrix(Z_ur);
}
/* ---------------------------------------------------------------------- */
SpMatrix Fuses2::compute_Q_minus_Lambda(const Matrix &Y) const {
  int r = Y.cols();
  int n = sp_.get_n();
  int dim = Q_.rows();      // dimension of data matrix Q_

  SpMatrix Lambda = Q_.selfadjointView<Eigen::Upper>();
  Lambda.reserve(dim);

  // try not to perform block operation on SparseMatrix Q_
  // this is too expensive: form very large matrix
  // Matrix symBlockDiag = Q_.selfadjointView<Eigen::Upper>()*Y*Y.transpose();
  Matrix Q_Y = Q_.selfadjointView<Eigen::Upper>() * Y;

  #pragma omp parallel for
  // Compute the first n entries on the diagonal
  for(int i=0; i<dim; ++i) {
    Lambda.insert(i, i) = - Q_Y.row(i).dot( Y.row(i) ); // symBlockDiag(i, i);
  }
  Lambda.makeCompressed();
  return Lambda;
}
/* ---------------------------------------------------------------------- */
bool Fuses2::compute_Q_minus_Lambda_min_eig(
    const Matrix &Y, double &min_eigenvalue, Eigen::VectorXd &min_eigenvector,
    unsigned int max_iterations, double min_eigenvalue_nonnegativity_tolerance,
    unsigned int num_Lanczos_vectors) const {
  // Check if num_Lanczos_vectors is greater than the number of rows of Y_
  if(num_Lanczos_vectors > Y_.rows()) {
    num_Lanczos_vectors = Y_.rows();
  }

  // The matrix of interest is C = Q - SymBlockDiag(Q*Y*Y')
  SpMatrix C = compute_Q_minus_Lambda(Y);

  // Construct matrix operation object using the wrapper class DenseSymMatPro
  SparseSymMatProd<double> op(C);

  // Construct eigen solver object,
  // requesting the eigenvalue with largest magnitude
  Spectra::SymEigsSolver<double, Spectra::SELECT_EIGENVALUE::LARGEST_MAGN,
      SparseSymMatProd<double>>
    largest_magnitude_eigensolver(&op, 1, num_Lanczos_vectors);

  // Initialize and compute
  largest_magnitude_eigensolver.init();
  int num_converged = largest_magnitude_eigensolver.compute(
      max_iterations, 1e-5, Spectra::SELECT_EIGENVALUE::LARGEST_MAGN);

  // Check convergence and bail out if necessary
  if(num_converged != 1) {
    return false;
  }

  // Retrieve results
  double lambda_lm = largest_magnitude_eigensolver.eigenvalues()(0);

  if(lambda_lm < 0) {
    // The largest-magnitude eigenvalue is negative, and therefore also the
    // minimum eigenvalue, so just return this solution
    min_eigenvalue = lambda_lm;
    min_eigenvector = largest_magnitude_eigensolver.eigenvectors(1);
    min_eigenvector.normalize(); // Ensure that this is a unit vector
    return true;
  }

  // The largest-magnitude eigenvalue is positive, and is therefore the maximum
  // eigenvalue.  Therefore, after shifting the spectrum of Q - Lambda by
  // -lambda_lm (by forming S = Q - Lambda - lambda_max*I), the shifted
  // eigenvalue will be all negative; in particular, the largest-magnitude
  // eigenvalue of S is lambda_min - lambda_max, with corresponding eigenvector
  // v_min

  // The matrix of interest is now S = C - lambda_lm*I
  SpMatrix S = C;
  for(int i=0; i<C.rows(); ++i) {
    S.coeffRef(i, i) -= lambda_lm;
  }

  // Construct matrix operation object using the wrapper class DenseSymMatPro
  SparseSymMatProd<double> shifted_op(S);

  // Construct eigen solver object, requesting the eigenvalue with largest
  // magnitude, which is the minimum eigenvalue in this case
  Spectra::SymEigsSolver<double, Spectra::SELECT_EIGENVALUE::LARGEST_MAGN,
      SparseSymMatProd<double>>
    min_eigensolver(&shifted_op, 1, num_Lanczos_vectors);

  // If Y is a critical point of F, then Y is also in the null space of
  // Q - Lambda(Y), and therefore its cols are eigenvectors corresponding to
  // the eigenvalue 0. In the case that the relaxation is exact, this is the
  // *minimum* eigenvalue, and therefore the rows of Y are exactly the
  // eigenvectors that we're looking for.  On the other hand, if the relaxation
  // is *not* exact, then Q - Lambda(Y) has at least one strictly negative
  // eigenvalue, and the rows of Y are *unstable fixed points* for the Lanczos
  // iterations.  Thus, we will take a slightly "fuzzed" version of the first
  // row of Y as an initialization for the Lanczos iterations; this allows for
  // rapid convergence in the case that the relaxation is exact (since are
  // starting close to a solution), while simultaneously allowing the
  // iterations to escape from this fixed point in the case that the relaxation
  // is not exact.
  Eigen::VectorXd v0 = Y.col(0);
  Eigen::VectorXd perturbation(v0.size());
  perturbation.setRandom();
  perturbation.normalize();
  Eigen::VectorXd xinit =
      v0 + (.03 * v0.norm()) * perturbation; // Perturb v0 by ~3%

  // Use this to initialize the eigensolver
  min_eigensolver.init(xinit.data());

  // Now determine the relative precision required in the Lanczos method in
  // order to be able to estimate the smallest eigenvalue within an *absolute*
  // tolerance of 'min_eigenvalue_nonnegativity_tolerance'
  num_converged = min_eigensolver.compute(
      max_iterations, min_eigenvalue_nonnegativity_tolerance / lambda_lm,
      Spectra::SELECT_EIGENVALUE::LARGEST_MAGN);

  // Check convergence and bail out if necessary
  if(num_converged != 1) {
    return false;
  }

  min_eigenvector = min_eigensolver.eigenvectors(1);
  min_eigenvector.normalize(); // Ensure that this is a unit vector
  min_eigenvalue = min_eigensolver.eigenvalues()(0) + lambda_lm;
  return true;
}
/* ---------------------------------------------------------------------- */
bool Fuses2::escape_saddle(const Matrix &Y, double lambda_min,
    const Vector &v_min, double gradient_tolerance,
    double preconditioned_gradient_tolerance, Matrix &Yplus) {

  //  v_min is an eigenvector corresponding to a negative eigenvalue of Q -
  //* Lambda, so the KKT conditions for the semidefinite relaxation are not
  //* satisfied; this implies that Y is a saddle point of the rank-restricted
  //* semidefinite  optimization.  Fortunately, v_min can be used to compute a
  //* descent  direction from this saddle point, as described in Theorem 3.9
  //* of the paper "A Riemannian Low-Rank Method for Optimization over
  //* Semidefinite  Matrices with Block-Diagonal Constraints". Define the vector
  //* Xdot := e_{r+1} * v'; this is a tangent vector to the domain of the SDP
  //* and provides a direction of negative curvature

  // Function value at current iterate (saddle point)
  double FY = evaluate_objective(Y);

  // Relaxation rank at the NEXT level of the Riemannian Staircase, i.e. we
  // require that r = Y.rows() + 1
  unsigned int r = get_relaxation_rank();

  // Construct the corresponding representation of the saddle point X in the
  // next level of the Riemannian Staircase by adding a row of 0's
  Matrix Y_augmented = Matrix::Zero(Y.rows(), r);
  Y_augmented.leftCols(r - 1) = Y;

  Matrix Ydot = Matrix::Zero(Y.rows(), r);
  Ydot.rightCols<1>() = v_min;

  // Set the initial step length to 100 times the distance needed to
  // arrive at a trial point whose gradient is large enough to avoid
  // triggering the gradient norm tolerance stopping condition,
  // according to the local second-order model
  double alpha = 2 * 100 * gradient_tolerance / fabs(lambda_min);
  double alpha_min = 1e-16; // Minimum stepsize

  // Initialize line search
  bool escape_success = false;

  Matrix Ytest;
  do {
    alpha /= 2;

    // Retract along the given tangent vector using the given stepsize
    Ytest = retract(Y_augmented, alpha * Ydot);

    // Ensure that the trial point Xtest has a lower function value than
    // the current iterate Y, and that the gradient at Ytest is
    // sufficiently large that we will not automatically trigger the
    // gradient tolerance stopping criterion at the next iteration
    double FYtest = evaluate_objective(Ytest);
    Matrix grad_FYtest = Riemannian_gradient(Ytest);
    double grad_FYtest_norm = grad_FYtest.norm();

    if((FYtest < FY) && (grad_FYtest_norm > gradient_tolerance)) {
      if(options_.precon == None) {
        escape_success = true;
      } else {
        double preconditioned_grad_FYtest_norm =
            precondition(Ytest, grad_FYtest).norm();
        if (preconditioned_grad_FYtest_norm >
              preconditioned_gradient_tolerance) {
          escape_success = true;
        }
      }
    }
  } while(!escape_success && (alpha > alpha_min));

  if(escape_success) {
    // Update initialization point for next level in the Staircase
    Yplus = Ytest;
    return true;
  } else {
    // If control reaches here, we exited the loop without finding a suitable
    // iterate, i.e. we failed to escape the saddle point
    return false;
  }
}
/* ---------------------------------------------------------------------- */
void Fuses2::solve() {
  /// ALGORITHM START
  auto fuses_start_time = Timer::tic();

  // Set number of threads // TODO: later we may enable multi threading
  //#ifdef USE_OMP
  //  omp_set_num_threads(options.num_threads);
  //#endif

  // ============== SET UP OPTIMIZATION =============
  /// Function handles required by the TNT optimization algorithm
  // Objective
  Optimization::Objective<Matrix, Matrix, std::vector<Matrix>> F =
      [this](const Matrix &Y, const Matrix &NablaF_Y,
      const std::vector<Matrix> &iterates) {
    return evaluate_objective(Y);
  };

  // Local quadratic model constructor
  Optimization::Smooth::QuadraticModel<Matrix, Matrix, Matrix,
      std::vector<Matrix>> QM =
      [this](const Matrix &Y, Matrix &grad,
      Optimization::Smooth::LinearOperator<Matrix, Matrix, Matrix,
          std::vector<Matrix>> &HessOp, Matrix &NablaF_Y,
      const std::vector<Matrix> &iterates) {

    // Compute and cache Euclidean gradient at the current iterate
    NablaF_Y = Euclidean_gradient(Y);

    // Compute Riemannian gradient from Euclidean gradient
    grad = Riemannian_gradient(Y, NablaF_Y);

    // Define linear operator for computing Riemannian Hessian-vector
    HessOp = [this](const Matrix &Y, const Matrix &Ydot, const Matrix &NablaF_Y,
        const std::vector<Matrix> &iterates) {
      return Riemannian_Hessian_vector_product(Y, NablaF_Y, Ydot);
    };
  };

  // Riemannian metric
  // We consider a realization of the product of Stiefel manifolds as an
  // embedded submanifold of R^{(n+k) x r}; consequently, the induced
  // Riemannian metric is simply the usual Euclidean inner product
  Optimization::Smooth::RiemannianMetric<Matrix, Matrix, Matrix,
      std::vector<Matrix>> metric = [this](const Matrix &Y, const Matrix &V1,
      const Matrix &V2, const Matrix &NablaF_Y,
      const std::vector<Matrix> &iterates) {
    return (V1 * V2.transpose()).trace();
  };

  // Retraction operator
  Optimization::Smooth::Retraction<Matrix, Matrix, Matrix, std::vector<Matrix>>
      retraction = [this](const Matrix &Y, const Matrix &Ydot,
      const Matrix &NablaF_Y, const std::vector<Matrix> &iterates) {
    return retract(Y, Ydot);
  };

  // Preconditioning operator (optional)
  std::experimental::optional<Optimization::Smooth::LinearOperator<Matrix,
    Matrix, Matrix, std::vector<Matrix>>> precon;
  if(options_.precon == None) {
    precon = std::experimental::nullopt;
  }
  else {
    Optimization::Smooth::LinearOperator<Matrix, Matrix, Matrix,
    std::vector<Matrix>>
    precon_op = [this](const Matrix &Y, const Matrix &Ydot,
        const Matrix &NablaF_Y,
        const std::vector<Matrix> &iterates) {
      return precondition(Y, Ydot);
    };
    precon = precon_op;
  }

  // Stat function (optional) -- used to record the sequence of iterates
  // computed during the Riemannian Staircase
  std::experimental::optional<Optimization::Smooth::TNTUserFunction<
      Matrix, Matrix, Matrix, std::vector<Matrix>>>
      user_function;

  if(options_.log_iterates) {
    Optimization::Smooth::TNTUserFunction<Matrix, Matrix, Matrix,
                                          std::vector<Matrix>>
        user_function_op =
            [](double t, const Matrix &Y, double f, const Matrix &g,
               const Optimization::Smooth::LinearOperator<
                   Matrix, Matrix, Matrix, std::vector<Matrix>> &HessOp,
               double Delta, unsigned int num_STPCG_iters, const Matrix &h,
               double df, double rho, bool accepted, const Matrix &NablaF_Y,
               std::vector<Matrix> &iterates) { iterates.push_back(Y); };
    user_function = user_function_op;
  } else {
    user_function = std::experimental::nullopt;
  }

  result_.timing.settingTNTHandles = Timer::toc(fuses_start_time);
  if(options_.verbose) {
    cout << "Done setting up optimization function handles. Time elapsed = "
         << result_.timing.settingTNTHandles << "s.\n" << endl;
  }

  // ============== RIEMANNIAN STAIRCASE =============
  if(options_.r0 > options_.rmax) {
    cout <<  "Could not start Riemannian staircase" << endl;
    throw runtime_error("rmax LESS THAN r0. (Try increasing rmax)");
  }

  // Configure optimization parameters
  Optimization::Smooth::TNTParams params;
  params.gradient_tolerance = options_.grad_norm_tol;
  params.preconditioned_gradient_tolerance =
      options_.preconditioned_grad_norm_tol;
  params.relative_decrease_tolerance = options_.rel_func_decrease_tol;
  params.stepsize_tolerance = options_.stepsize_tol;
  params.max_iterations = options_.max_iterations;
  params.max_TPCG_iterations = options_.max_tCG_iterations;
  params.verbose = options_.verbose;

  auto riemannian_staircase_start_time = Timer::tic();
  if(options_.verbose) {
    cout << "Begin Riemannian staircase" << endl;
  }
  for(size_t r = options_.r0; r <= options_.rmax; r++) {

    // The elapsed time from the start of the Riemannian Staircase algorithm
    // until the start of this iteration of RTR
    double RTR_iteration_accum_time =
        Timer::toc(riemannian_staircase_start_time);

    // Test temporal stopping condition
    if(RTR_iteration_accum_time >= options_.max_computation_time) {
      result_.status = ELAPSED_TIME;
      break;
    }

    // Set maximum permitted computation time for this level of the Riemannian
    // Staircase
    params.max_computation_time =
        options_.max_computation_time - RTR_iteration_accum_time;

    std::vector<gtsam::Matrix> resultIterates; // for logging

    /// Run optimization!
    Optimization::Smooth::TNTResult<Matrix> TNTResults =
        Optimization::Smooth::TNT<Matrix, Matrix, Matrix, std::vector<Matrix>>
        (F, QM, metric, retraction, Y_, NablaF_Y_, resultIterates, precon,
        params, user_function);

    if(options_.log_iterates) {
      result_.iterates.push_back(resultIterates);
    }

    // Extract the results
    result_.Yopt = TNTResults.x;
    result_.SDPval = TNTResults.f;
    result_.gradnorm = Riemannian_gradient(result_.Yopt).norm();

    // Record sequence of function values
    result_.function_values.push_back(TNTResults.objective_values);

    // Record sequence of gradient norm values
    result_.gradient_norms.push_back(TNTResults.gradient_norms);

    // Record sequence of elapsed optimization times
    result_.timing.elapsed_optimization.push_back(TNTResults.time);

    if(options_.verbose) {
      cout << fixed << "  Current rank = " << r
           << ", optimal value = " << evaluate_objective(result_.Yopt)
           << ", time elapsed on Riemannian staircase = "
           << Timer::toc(riemannian_staircase_start_time) * 1000
           << " ms." << endl;
    }

    /// Check second-order optimality
    // Compute the minimum eigenvalue lambda and corresponding eigenvector
    // of Q - Lambda
    auto eig_start_time = Timer::tic();
    bool eigenvalue_convergence = compute_Q_minus_Lambda_min_eig(
        result_.Yopt, result_.lambda_min, result_.v_min,
        options_.max_eig_iterations, options_.min_eig_num_tol,
        options_.num_Lanczos_vectors);
    double eig_elapsed_time = Timer::toc(eig_start_time);
    result_.timing.minimum_eigenvalue_computation.push_back(
            eig_elapsed_time);

    // Check eigenvalue convergence
    if(!eigenvalue_convergence) {
      std::cout << "WARNING!  EIGENVALUE COMPUTATION DID NOT CONVERGE TO "
                   "DESIRED PRECISION!" << std::endl;
      result_.status = EIG_IMPRECISION;
      break;
    }

    // Record results of eigenvalue computation
    result_.minimum_eigenvalues.push_back(result_.lambda_min);

    // Test nonnegativity of minimum eigenvalue
    if(result_.lambda_min > -options_.min_eig_num_tol) {
      // results.Yopt is a second-order critical point (global optimum)!
      if (options_.verbose) {
        cout << "Found second-order critical point! (minimum eigenvalue = "
             << result_.lambda_min << "). Time elapsed = "
             << eig_elapsed_time * 1000 << " ms\n" << endl;
      }
      result_.status = GLOBAL_OPT;
      break;
    } // global optimality
    else {
      if (options_.verbose) {
        cout << "Saddle point with minimum eigenvalue = " << result_.lambda_min
             << endl;
      }
      /// ESCAPE FROM SADDLE!
      // Augment the rank of the rank-restricted semidefinite relaxation in
      // preparation for ascending to the next level of the Riemannian Staircase
      set_relaxation_rank(r + 1);
      Matrix Yplus;
      bool escape_success = escape_saddle(result_.Yopt, result_.lambda_min,
          result_.v_min, options_.grad_norm_tol,
          options_.preconditioned_grad_norm_tol, Yplus);

      if(escape_success) {
        // Update initialization point for next level in the Staircase
        Y_ = Yplus;
//        std::cout << "WARNING! ENTERING SECOND RIEMANNIAN STAIRCASE!"
//            << std::endl;
      } else {
        std::cout << "WARNING!  BACKTRACKING LINE SEARCH FAILED TO ESCAPE FROM "
                     "SADDLE POINT!  (Try decreasing the preconditioned "
                     "gradient norm tolerance)" << std::endl;
        result_.status = SADDLE_POINT;
        break;
      }
    } // saddle point
  }   // Riemannian Staircase

  // ============== POST-PROCESSING =============
  // Evaluate objective function at ROUNDED solution
  auto myTime = Timer::tic();
  result_.xhat = round_solution(result_.Yopt);
  result_.timing.rounding = Timer::toc(myTime);
  result_.Fxhat = crf_.evaluate(result_.xhat);

  // search if we found a better solution in past iterations
  if(options_.use_best_rounded && options_.log_iterates){
    // for each step of the staircase
    for(size_t step = 0; step<result_.iterates.size(); step++){
      // for each iterate
      for(size_t it = 0; it<result_.iterates[step].size(); it++){
        auto myTime = Timer::tic();
        Matrix X = round_solution(result_.iterates[step][it]);
        result_.timing.rounding += Timer::toc(myTime);
        // check if past solution was better
        double objectiveRounded = crf_.evaluate(X);
        if(objectiveRounded < result_.Fxhat){
          // we found a better solution, let's store it
          result_.Fxhat = objectiveRounded;
          result_.xhat = X;
        }
      }
    }
  }

  // finish counting time
  result_.timing.total = Timer::toc(fuses_start_time);

  if(options_.verbose) {
    // print objective function values
    cout << "  Optimal value before rounding = "
         << evaluate_objective(result_.Yopt) << endl;
    cout << "  Optimal value after rounding = " << result_.Fxhat << endl;
    cout << "Done solving the optimization problem. Total time elapsed = "
         << result_.timing.total * 1000 << " ms.\n" << endl;
  }
}
/* ---------------------------------------------------------------------- */
} // namespace Fuses2

