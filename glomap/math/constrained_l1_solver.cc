#include "glomap/math/constrained_l1_solver.h"

#include <colmap/util/logging.h>

namespace HETA {

ConstrainedL1Solver::ConstrainedL1Solver(const Options& options,
                                         const Eigen::SparseMatrix<double>& A,
                                         const Eigen::VectorXd& b,
                                         const int num_l1_residuals,
                                         const int num_constraints)
    : options_(options),
      A_(A),
      b_(b),
      num_l1_residuals_(num_l1_residuals),
      num_inequality_constraints_(num_constraints) {
  const Eigen::SparseMatrix<double> spd_mat = A_.transpose() * A_;
  linear_solver_.compute(spd_mat);
  CHECK_EQ(num_l1_residuals + num_constraints, A_.rows());
  CHECK_EQ(linear_solver_.info(), Eigen::Success);
}

// We create a modified L1 solver such that ||Bx - b|| is minimized under L1
// norm subject to the constraint geq_mat * x > geq_vec. We conveniently
// create this constraint in ADMM terms as:
//
//    minimize f(x) + g(z_1) + h(z_2)
//    s.t. Bx - b - z_1 = 0
//         Cx - c - z_2 = 0
//
// Where f(x) = 0, g(z_1) = |z_1| and h(z_2) is an indicate function for our
// inequality constraint. This can be transformed into the standard ADMM
// formulation as:
//
//    minimize f(x) + g(z)
//    s.t. A * x - d - z = 0
//
// where A = [B;C] and d=[b;c] (where ; is the "stack" operation like matlab)
// This can now be solved in the same form as the L1 minimization, with a
// slightly different z update.
void ConstrainedL1Solver::FastAdapSolve(Eigen::VectorXd* solution) {
  CHECK_NOTNULL(solution)->resize(A_.cols());
  Eigen::VectorXd& x = *solution;

  Eigen::VectorXd z = Eigen::VectorXd::Zero(A_.rows());
  Eigen::VectorXd u = Eigen::VectorXd::Zero(A_.rows());
  Eigen::VectorXd z_plus_b = z + b_;
  Eigen::VectorXd a_times_x(A_.rows());
  Eigen::VectorXd z_old(A_.rows());
  Eigen::VectorXd a_times_x_minus_z_plus_b(A_.rows());
  Eigen::VectorXd residual(A_.rows());

  // Precompute some convergence terms.
  const double rhs_norm = b_.norm();
  const double primal_abs_tolerance_eps =
      std::sqrt(A_.rows()) * options_.absolute_tolerance;
  const double dual_abs_tolerance_eps =
      std::sqrt(A_.cols()) * options_.absolute_tolerance;

  Eigen::SparseMatrix<double> A_transpose = A_.transpose();
  VLOG(2) << "Start ADMM Solve";

  // int constrain_num = A_.rows(), constrain_num_old = constrain_num;
  for (int i = 0; i <= options_.max_num_iterations; i++) {
    x.noalias() = linear_solver_.solve(A_transpose * (z_plus_b - u));
    a_times_x.noalias() = A_ * x;
    // std::swap(z, z_old);

    residual = a_times_x - b_ + u;

    // This method is used for the z-update, which is conveniently an
    // element-wise update. For the terms in vec corresponding to the L1
    // minimization, we update the values with the L1 proximal mapping
    // (Shrinkage) operator. The terms corresponding to the inequality
    // constraints are constrained to be greater than zero as vec = max(vec, 0).
    z.head(num_l1_residuals_).array() =
        (residual.head(num_l1_residuals_).array() - 1.0 / rho_).max(0.0) -
        (-residual.head(num_l1_residuals_).array() - 1.0 / rho_).max(0.0);

    z.tail(num_inequality_constraints_).array() =
        residual.tail(num_inequality_constraints_).array().max(0.0);

    z_plus_b = z + b_;

    a_times_x_minus_z_plus_b = a_times_x - z_plus_b;

    // Update u.
    u.noalias() += a_times_x_minus_z_plus_b;
    const double z_norm = z.norm();

    if ((i + 1) % 20 == 0) {
      // Compute the convergence terms.
      const double r_norm = a_times_x_minus_z_plus_b.norm();
      const double a_times_x_norm = a_times_x.norm();

      const double primal_eps =
          primal_abs_tolerance_eps +
          options_.relative_tolerance *
              std::max({a_times_x_norm, z_norm, rhs_norm});

      // Relax stopping condition for efficiency
      if (r_norm < primal_eps) {
        VLOG(2) << "Iteratate times: " << i + 1;
        break;
      }
    }

    // Adaptive rho for acceleration
    const double new_rho = std::clamp(
        u.norm() / z_norm * rho_, options_.rho_min, options_.rho_max);
    rho_ = (1 - omega_) * rho_ + omega_ * new_rho;
    omega_ *= options_.omega_scale;
  }
}

}  // namespace HETA
