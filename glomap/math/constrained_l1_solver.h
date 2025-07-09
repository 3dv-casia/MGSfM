#pragma once

#include <Eigen/Cholesky>
#include <Eigen/CholmodSupport>
#include <Eigen/Core>

namespace HETA {

class ConstrainedL1Solver {
 public:
  struct Options {
    int max_num_iterations = 1000;
    // Rho is the augmented Lagrangian parameter for the L1 minimization.
    double rho = 10.0;  // 10.0
    // Alpha is the over-relaxation parameters (typically between 1.0
    // and 1.8).
    double alpha = 1.2;  // 1.2

    // Stopping criteria.
    double absolute_tolerance = 1e-4 * 0.3;
    double relative_tolerance = 1e-2 * 0.3;

    double rho_min = 1e-4;
    double rho_max = 1e4;
    double omega_scale = std::pow(2, -0.01);
  };

  // The linear system along with the equality and inequality constraints.
  ConstrainedL1Solver(const Options& options,
                      const Eigen::SparseMatrix<double>& A,
                      const Eigen::VectorXd& b,
                      const int num_l1_residuals,
                      const int num_constraints);

  // Solve the constrained L1 minimization above.
  void FastAdapSolve(Eigen::VectorXd* solution);

 private:
  const Options options_;

  const Eigen::SparseMatrix<double>& A_;
  const Eigen::VectorXd& b_;
  const int num_l1_residuals_;
  const int num_inequality_constraints_;

  // Matrix A where || Ax - b ||_1 is the problem we are solving.

  double rho_ = 10.0;
  double omega_ = 1.0;

  // Cholesky linear solver. Since our linear system will be a SPD matrix we
  // can utilize the Cholesky factorization.
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> linear_solver_;
};

}  // namespace HETA
