// This code is adapted from Theia library (http://theia-sfm.org/),
// with its original L1 solver adapted from
//  "https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html"

#pragma once

#include <colmap/util/logging.h>

#include <Eigen/Core>

namespace HETA {
struct L1SolverVec3Options {
  int max_num_iterations = 1000;
  // Rho is the augmented Lagrangian parameter.
  double rho = 1.0;
  // Alpha is the over-relaxation parameter (typically between 1.0 and 1.8).
  double alpha = 1.0;

  double absolute_tolerance = 1e-4;
  double relative_tolerance = 1e-2;
};

class L1SolverVec3 {
 public:
  L1SolverVec3(const L1SolverVec3Options& options, const Eigen::MatrixX3d& mat)
      : options_(options), a_(mat) {
    const Eigen::Matrix3d full_mat = a_.transpose() * a_;
    linear_solver_.compute(full_mat);
    CHECK_EQ(linear_solver_.info(), Eigen::Success);
  }

  void Solve(const Eigen::VectorXd& rhs, Eigen::Vector3d* solution) {
    Eigen::Vector3d& x = *solution;

    Eigen::VectorXd z = Eigen::VectorXd::Zero(a_.rows());
    Eigen::VectorXd u = Eigen::VectorXd::Zero(a_.rows());
    Eigen::ArrayXd residual(a_.rows());

    Eigen::VectorXd a_times_x(a_.rows()), z_old(z.size()), ax_hat(a_.rows());
    // Precompute some convergence terms.
    const double rhs_norm = rhs.norm();
    const double primal_abs_tolerance_eps =
        std::sqrt(a_.rows()) * options_.absolute_tolerance;
    const double dual_abs_tolerance_eps =
        std::sqrt(a_.cols()) * options_.absolute_tolerance;

    for (int i = 0; i < options_.max_num_iterations; i++) {
      // Update x.
      x.noalias() = linear_solver_.solve(a_.transpose() * (rhs + z - u));

      a_times_x.noalias() = a_ * x;
      ax_hat.noalias() = options_.alpha * a_times_x;
      ax_hat.noalias() += (1.0 - options_.alpha) * (z + rhs);

      // Update z and set z_old.
      std::swap(z, z_old);
      residual = (ax_hat - rhs + u).array();
      z.noalias() = ((residual - 1.0 / options_.rho).max(0.0) -
                     (-residual - 1.0 / options_.rho).max(0.0))
                        .matrix();

      // Update u.
      u.noalias() += ax_hat - z - rhs;

      // Compute the convergence terms.
      const double r_norm = (a_times_x - z - rhs).norm();
      const double s_norm =
          (-options_.rho * a_.transpose() * (z - z_old)).norm();
      const double max_norm = std::max({a_times_x.norm(), z.norm(), rhs_norm});
      const double primal_eps =
          primal_abs_tolerance_eps + options_.relative_tolerance * max_norm;
      const double dual_eps = dual_abs_tolerance_eps +
                              options_.relative_tolerance *
                                  (options_.rho * a_.transpose() * u).norm();

      // Determine if the minimizer has converged.
      if (r_norm < primal_eps && s_norm < dual_eps) {
        break;
      }
    }
  }

 private:
  const L1SolverVec3Options& options_;

  // Matrix A in || Ax - b ||_1
  const Eigen::MatrixX3d a_;

  Eigen::LLT<Eigen::Matrix3d> linear_solver_;
};

}  // namespace HETA
