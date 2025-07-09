#pragma once

#include "glomap/math/l1_solver.h"
#include "glomap/scene/types_sfm.h"
#include "glomap/types.h"

#include <colmap/math/math.h>
#include <colmap/math/random.h>

#include <string>
#include <vector>

// Code is adapted from Theia's RobustRotationEstimator
// (http://www.theia-sfm.org/). For gravity aligned rotation averaging, refere
// to the paper "Gravity Aligned Rotation Averaging"
namespace MGSfM {
using namespace glomap;
// The struct to store the temporary information for each image pair
struct RefImagePairInfo {
  RefImagePairInfo()
      : pair_id(-1),
        ref_image_id1(-1),
        ref_image_id2(-1),
        num_inliers(0),
        ref_rel_rotation(Eigen::Quaterniond::Identity()) {}

  RefImagePairInfo(image_pair_t pair_id,
                   image_t image_id1,
                   image_t image_id2,
                   uint32_t num_inliers,
                   Eigen::Quaterniond ref_rel_rotation)
      : pair_id(pair_id),
        ref_image_id1(image_id1),
        ref_image_id2(image_id2),
        num_inliers(num_inliers),
        ref_rel_rotation(ref_rel_rotation) {}

  image_pair_t pair_id;  // raw pair id in the view graph

  image_t ref_image_id1, ref_image_id2;

  uint32_t num_inliers;

  Eigen::Quaterniond ref_rel_rotation;
};

struct MultiRotationEstimatorOptions {
  // Maximum number of times to run L1 minimization.
  int max_num_l1_iterations = 5;

  // Average step size threshold to terminate the L1 minimization
  double l1_step_convergence_threshold = 0.001;

  // The number of iterative reweighted least squares iterations to perform.
  int max_num_irls_iterations = 100;

  // Average step size threshold to termininate the IRLS minimization
  double irls_step_convergence_threshold = 0.001;

  Eigen::Vector3d axis = Eigen::Vector3d(0, 1, 0);

  // This is the point where the Huber-like cost function switches from L1 to
  // L2.
  double irls_loss_parameter_sigma = 0.5;  // in degree

  enum WeightType {
    // For Geman-McClure weight, refer to the paper "Efficient and robust
    // large-scale rotation averaging" (Chatterjee et. al, 2013)
    GEMAN_MCCLURE,
    // For Half Norm, refer to the paper "Robust Relative Rotation Averaging"
    // (Chatterjee et. al, 2017)
    HALF_NORM,
  } weight_type = HALF_NORM;

  // Flg to use maximum spanning tree for initialization
  bool skip_initialization = false;

  // TODO: Implement the weighted version for rotation averaging
  bool use_weight = false;
};

class MultiRotationEstimator {
 public:
  static Eigen::Quaterniond MedianRotations(
      const std::vector<Eigen::Quaterniond>& rotations,
      std::vector<double>& final_angle_errors) {
    CHECK(!rotations.empty());
    constexpr int times = 10;
    constexpr int kMaxIterations = 1e5;
    constexpr double kEpsilon = 1e-5;
    std::vector<int> all_random_ids(times);
    for (int i = 0; i < times; ++i) {
      all_random_ids[i] = colmap::RandomUniformInteger(
          0, static_cast<int>(rotations.size() - 1));
    }
    const int n = rotations.size();
    Eigen::Quaterniond best_median_rotation;
    double min_median_error = std::numeric_limits<double>::max();

#pragma omp parallel for schedule(dynamic)
    for (const auto& id : all_random_ids) {
      Eigen::Quaterniond median_rotation = rotations.at(id);

      std::vector<double> residuals_0(n), residuals_1(n), residuals_2(n);
      std::vector<double> angle_errors(n);
      double median_error;

      for (int iter = 0; iter < kMaxIterations; ++iter) {
        const Eigen::Quaterniond inv_median = median_rotation.inverse();
        for (int i = 0; i < n; ++i) {
          Eigen::AngleAxisd residual(rotations.at(i) * inv_median);
          Eigen::Vector3d residual_vec = residual.angle() * residual.axis();
          residuals_0[i] = residual_vec[0];
          residuals_1[i] = residual_vec[1];
          residuals_2[i] = residual_vec[2];
          angle_errors[i] = residual.angle();
        }

        const double med0 = colmap::Median(residuals_0);
        const double med1 = colmap::Median(residuals_1);
        const double med2 = colmap::Median(residuals_2);
        Eigen::Vector3d median_residual_vec(med0, med1, med2);

        const double norm = median_residual_vec.norm();
        if (norm < kEpsilon) {
          median_error = colmap::Median(angle_errors);
#pragma omp critical
          if (median_error < min_median_error) {
            best_median_rotation = median_rotation;
            min_median_error = median_error;
            final_angle_errors.swap(angle_errors);
          }
          break;
        }

        Eigen::AngleAxisd update(norm, median_residual_vec.normalized());
        median_rotation =
            (Eigen::Quaterniond(update) * median_rotation).normalized();
      }
    }
    return best_median_rotation;
  }
  using RigIdPair = std::pair<image_t, image_t>;
  explicit MultiRotationEstimator(const MultiRotationEstimatorOptions& options)
      : options_(options) {}

  // Estimates the global orientations of all views based on an initial
  // guess. Returns true on successful estimation and false otherwise.
  bool EstimateRotations(
      const ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      const std::unordered_map<camera_t, Rigid3d>& rel_from_refs,
      const std::unordered_map<image_t, image_t>& image_ref_id);

  MultiRotationEstimatorOptions& GetOptions() { return options_; }

 protected:
  // Initialize the rotation from the maximum spanning tree
  // Number of inliers serve as weights
  void InitializeFromRigMaximumSpanningTree(
      const ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      const std::unordered_set<image_t>& ref_image_ids,
      const std::vector<RefImagePairInfo>& rig_ref_image_pairs);

  // Sets up the sparse linear system such that dR_ij = dR_j - dR_i. This is
  // the first-order approximation of the angle-axis rotations. This should
  // only be called once.
  void SetupLinearSystem(std::unordered_map<image_t, Image>& images,
                         const std::unordered_set<image_t>& ref_image_ids,
                         const std::vector<RefImagePairInfo>& ref_image_pairs);

  // Performs the L1 robust loss minimization.
  bool SolveL1Regression(const ViewGraph& view_graph,
                         std::unordered_map<image_t, Image>& images,
                         const std::unordered_set<image_t>& ref_image_ids,
                         const std::vector<RefImagePairInfo>& ref_image_pairs);

  // Performs the iteratively reweighted least squares.
  bool SolveIRLS(const ViewGraph& view_graph,
                 std::unordered_map<image_t, Image>& images,
                 const std::unordered_set<image_t>& ref_image_ids,
                 const std::vector<RefImagePairInfo>& ref_image_pairs);

  // Updates the global rotations based on the current rotation change.
  void UpdateGlobalRotations(std::unordered_map<image_t, Image>& images,
                             const std::unordered_set<image_t>& ref_image_ids);

  // Computes the relative rotation (tangent space) residuals based on the
  // current global orientation estimates.
  void ComputeResiduals(const ViewGraph& view_graph,
                        std::unordered_map<image_t, Image>& images,
                        const std::vector<RefImagePairInfo>& ref_image_pairs);

  // Computes the average size of the most recent step of the algorithm.
  // The is the average over all non-fixed global_orientations_ of their
  // rotation magnitudes.
  double ComputeAverageStepSize(
      const std::unordered_map<image_t, Image>& images,
      const std::unordered_set<image_t>& ref_image_ids);

  // Data
  // Options for the solver.
  MultiRotationEstimatorOptions options_;

  // The sparse matrix used to maintain the linear system. This is matrix A in
  // Ax = b.
  Eigen::SparseMatrix<double> sparse_matrix_;

  // x in the linear system Ax = b.
  Eigen::VectorXd tangent_space_step_;

  // b in the linear system Ax = b.
  Eigen::VectorXd tangent_space_residual_;

  Eigen::VectorXd rotation_estimated_;

  // Varaibles for intermidiate results
  std::unordered_map<image_t, image_t> image_id_to_idx_;

  // The fixed camera id. This is used to remove the ambiguity of the linear
  image_t fixed_camera_id_ = -1;

  // The fixed camera rotation (if with initialization, it would not be identity
  // matrix)
  Eigen::Vector3d fixed_camera_rotation_;
};

}  // namespace MGSfM
