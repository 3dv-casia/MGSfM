#include "multi_global_rotation_averaging.h"

#include "glomap/math/l1_solver.h"
#include "glomap/math/rigid3d.h"
#include "glomap/math/tree.h"

#include <colmap/math/math.h>

#include <iostream>
#include <queue>

namespace MGSfM {
using namespace glomap;
bool MultiRotationEstimator::EstimateRotations(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_map<camera_t, Rigid3d>& rel_from_refs,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  std::vector<RefImagePairInfo> all_rig_pairs;
  all_rig_pairs.reserve(view_graph.image_pairs.size());
  std::unordered_set<image_t> ref_image_ids;

  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    if (image_ref_id.at(image_pair.image_id1) ==
        image_ref_id.at(image_pair.image_id2)) {
      continue;
    }
    image_t ref_image_id1 = image_ref_id.at(image_pair.image_id1);
    image_t ref_image_id2 = image_ref_id.at(image_pair.image_id2);
    ref_image_ids.emplace(ref_image_id1);
    ref_image_ids.emplace(ref_image_id2);

    camera_t camera_id1 = images.at(image_pair.image_id1).camera_id;
    camera_t camera_id2 = images.at(image_pair.image_id2).camera_id;
    Eigen::Quaterniond rel_rig_rotation =
        rel_from_refs.at(camera_id2).rotation.inverse() *
        image_pair.cam2_from_cam1.rotation *
        rel_from_refs.at(camera_id1).rotation;

    all_rig_pairs.emplace_back(RefImagePairInfo(pair_id,
                                                ref_image_id1,
                                                ref_image_id2,
                                                image_pair.inliers.size(),
                                                rel_rig_rotation));
  }

  // Initialize the rotation from maximum spanning tree
  if (!options_.skip_initialization) {
    // Cauculate all relative rotations between reference cameras
    std::unordered_map<RigIdPair, std::vector<RefImagePairInfo>> rig_pairs_map;

    for (const auto& pair : all_rig_pairs) {
      RigIdPair image_id_pair;
      if (pair.ref_image_id1 < pair.ref_image_id2) {
        image_id_pair = std::make_pair(pair.ref_image_id1, pair.ref_image_id2);
        rig_pairs_map[image_id_pair].emplace_back(pair);
      } else {
        image_id_pair = std::make_pair(pair.ref_image_id2, pair.ref_image_id1);
        auto new_pair = pair;
        new_pair.ref_image_id1 = pair.ref_image_id2;
        new_pair.ref_image_id2 = pair.ref_image_id1;
        new_pair.ref_rel_rotation = pair.ref_rel_rotation.inverse();
        rig_pairs_map[image_id_pair].emplace_back(new_pair);
      }
    }

    // Compute the median relative rotation and the number of inlier matches by
    // considering relative rotations with angular errors less than 5 degrees.
    std::vector<RefImagePairInfo> median_rig_pairs;
    median_rig_pairs.reserve(rig_pairs_map.size());
    for (const auto& [image_id_pair, pair_infos] : rig_pairs_map) {
      const size_t pair_infos_num = pair_infos.size();
      std::vector<Eigen::Quaterniond> rotations;
      rotations.reserve(pair_infos_num);
      int max_index = 0;
      int max_num_inliers = pair_infos.at(0).num_inliers;
      for (int i = 0; i < pair_infos_num; ++i) {
        auto& pair_info = pair_infos.at(i);
        rotations.emplace_back(pair_info.ref_rel_rotation);
        if (pair_info.num_inliers > max_num_inliers) {
          max_index = i;
          max_num_inliers = pair_info.num_inliers;
        }
      }
      std::vector<double> angle_errors;
      Eigen::Quaterniond median_rotation =
          MedianRotations(rotations, angle_errors);
      uint32_t all_inlier_number = 0;
      if (angle_errors.size() == pair_infos_num)
        for (int i = 0; i < pair_infos_num; i++) {
          if (angle_errors.at(i) < DegToRad(5.0))
            all_inlier_number += pair_infos.at(i).num_inliers;
        }
      else
        median_rotation = pair_infos.at(max_index).ref_rel_rotation;
      auto rig_pair_info = pair_infos.front();
      rig_pair_info.pair_id = -1;
      rig_pair_info.ref_rel_rotation = median_rotation;
      rig_pair_info.num_inliers = all_inlier_number;
      median_rig_pairs.emplace_back(rig_pair_info);
    }

    // Initialize the global rotations of reference images
    InitializeFromRigMaximumSpanningTree(
        view_graph, images, ref_image_ids, median_rig_pairs);

    // Filter relative rotations with angular error larger than 15 degrees
    std::vector<RefImagePairInfo> new_rig_pairs;
    new_rig_pairs.reserve(all_rig_pairs.size());
    for (const auto& ref_image_pair : all_rig_pairs) {
      Eigen::Quaterniond residual_rotation =
          ref_image_pair.ref_rel_rotation *
          images.at(ref_image_pair.ref_image_id1).cam_from_world.rotation *
          images.at(ref_image_pair.ref_image_id2)
              .cam_from_world.rotation.conjugate();

      double cos_r = (residual_rotation.toRotationMatrix().trace() - 1) / 2;
      double angle = RadToDeg(std::acos(std::min(std::max(cos_r, -1.), 1.)));
      if (angle > 15.0) continue;
      new_rig_pairs.emplace_back(ref_image_pair);
    }
    all_rig_pairs.swap(new_rig_pairs);
  }

  // Set up the linear system
  SetupLinearSystem(images, ref_image_ids, all_rig_pairs);

  // Solve the linear system for L1 norm optimization
  if (options_.max_num_l1_iterations > 0) {
    if (!SolveL1Regression(view_graph, images, ref_image_ids, all_rig_pairs)) {
      return false;
    }
  }

  // Solve the linear system for IRLS optimization
  if (options_.max_num_irls_iterations > 0) {
    if (!SolveIRLS(view_graph, images, ref_image_ids, all_rig_pairs)) {
      return false;
    }
  }

  // Convert the final results
  for (auto& [image_id, image] : images) {
    if (!image.is_registered) continue;
    image.cam_from_world.rotation =
        rel_from_refs.at(image.camera_id).rotation *
        Eigen::Quaterniond(AngleAxisToRotation(rotation_estimated_.segment(
            image_id_to_idx_[image_ref_id.at(image_id)], 3)));
  }

  return true;
}

void MultiRotationEstimator::InitializeFromRigMaximumSpanningTree(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_set<image_t>& ref_image_ids,
    const std::vector<RefImagePairInfo>& median_rig_pairs) {
  std::unordered_map<std::pair<image_t, image_t>, uint32_t> rig_pair_indices;
  uint32_t rig_pair_size = median_rig_pairs.size();
  for (int i = 0; i < rig_pair_size; i++) {
    const auto& rig_pair = median_rig_pairs.at(i);
    CHECK_LT(rig_pair.ref_image_id1, rig_pair.ref_image_id2);
    rig_pair_indices[std::make_pair(rig_pair.ref_image_id1,
                                    rig_pair.ref_image_id2)] = i;
  }

  // Here, we assume that largest connected component is already retrieved, so
  // we do not need to do that again compute maximum spanning tree.
  std::unordered_map<image_t, image_t> parents;
  image_t root = RigMaximumSpanningTree(
      view_graph, images, parents, ref_image_ids, median_rig_pairs);

  // Iterate through the tree to initialize the rotation
  // Establish child info
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& image_id : ref_image_ids) {
    const auto& image = images.at(image_id);
    if (!image.is_registered) continue;
    children.insert(std::make_pair(image_id, std::vector<image_t>()));
  }
  for (auto& [child, parent] : parents) {
    if (root == child) continue;
    children[parent].emplace_back(child);
  }

  std::queue<image_t> indexes;
  indexes.push(root);

  while (!indexes.empty()) {
    image_t curr = indexes.front();
    indexes.pop();

    // Add all children into the tree
    for (auto& child : children[curr]) indexes.push(child);
    // If it is root, then fix it to be the original estimation
    if (curr == root) continue;
    std::pair<image_t, image_t> image_id_pair =
        curr < parents[curr] ? std::make_pair(curr, parents[curr])
                             : std::make_pair(parents[curr], curr);
    const auto& rig_pair =
        median_rig_pairs.at(rig_pair_indices.at(image_id_pair));
    // Directly use the relative pose for estimation rotation
    if (rig_pair.ref_image_id1 == curr) {
      // 1_R_w = 2_R_1^T * 2_R_w
      images[curr].cam_from_world.rotation =
          rig_pair.ref_rel_rotation.inverse() *
          images[parents[curr]].cam_from_world.rotation;
    } else {
      // 2_R_w = 2_R_1 * 1_R_w
      images[curr].cam_from_world.rotation =
          rig_pair.ref_rel_rotation *
          images[parents[curr]].cam_from_world.rotation;
    }
  }
}

void MultiRotationEstimator::SetupLinearSystem(
    std::unordered_map<image_t, Image>& images,
    const std::unordered_set<image_t>& ref_image_ids,
    const std::vector<RefImagePairInfo>& all_rig_pairs) {
  // Clear all the structures
  sparse_matrix_.resize(0, 0);
  tangent_space_step_.resize(0);
  tangent_space_residual_.resize(0);
  rotation_estimated_.resize(0);
  image_id_to_idx_.clear();

  // Initialize the structures for estimated rotation
  image_id_to_idx_.reserve(images.size());
  // allocate more memory than needed
  rotation_estimated_.resize(3 * images.size());
  image_t num_dof = 0;
  for (const auto& image_id : ref_image_ids) {
    const auto& image = images.at(image_id);
    if (!image.is_registered) continue;
    image_id_to_idx_[image_id] = num_dof;

    rotation_estimated_.segment(num_dof, 3) =
        Rigid3dToAngleAxis(image.cam_from_world);
    num_dof += 3;
  }

  // If no cameras are set to be fixed, then take the first camera
  if (fixed_camera_id_ == -1) {
    for (const auto& image_id : ref_image_ids) {
      const auto& image = images.at(image_id);
      if (!image.is_registered) continue;
      fixed_camera_id_ = image_id;
      fixed_camera_rotation_ = Rigid3dToAngleAxis(image.cam_from_world);
      break;
    }
  }

  rotation_estimated_.conservativeResize(num_dof);

  std::vector<Eigen::Triplet<double>> coeffs;
  coeffs.reserve(all_rig_pairs.size() * 6 + 3);

  // Establish linear systems
  size_t curr_pos = 0;
  for (const auto& ref_image_pair : all_rig_pairs) {
    int image_id1 = ref_image_pair.ref_image_id1;
    int image_id2 = ref_image_pair.ref_image_id2;

    int vector_idx1 = image_id_to_idx_[image_id1];
    int vector_idx2 = image_id_to_idx_[image_id2];

    for (int i = 0; i < 3; i++) {
      coeffs.emplace_back(
          Eigen::Triplet<double>(curr_pos + i, vector_idx1 + i, -1));
      coeffs.emplace_back(
          Eigen::Triplet<double>(curr_pos + i, vector_idx2 + i, 1));
    }
    curr_pos += 3;
  }

  // Set some cameras to be fixed
  // if some cameras have gravity, then add a single term constraint
  // Else, change to 3 constriants

  for (int i = 0; i < 3; i++) {
    coeffs.emplace_back(Eigen::Triplet<double>(
        curr_pos + i, image_id_to_idx_[fixed_camera_id_] + i, 1));
  }
  curr_pos += 3;

  sparse_matrix_.resize(curr_pos, num_dof);
  sparse_matrix_.setFromTriplets(coeffs.begin(), coeffs.end());

  // Initialize x and b
  tangent_space_step_.resize(num_dof);
  tangent_space_residual_.resize(curr_pos);
}

bool MultiRotationEstimator::SolveL1Regression(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_set<image_t>& ref_image_ids,
    const std::vector<RefImagePairInfo>& all_rig_pairs) {
  L1SolverOptions opt_l1_solver;
  opt_l1_solver.max_num_iterations = 10;

  L1Solver<Eigen::SparseMatrix<double>> l1_solver(opt_l1_solver,
                                                  sparse_matrix_);
  double last_norm = 0;
  double curr_norm = 0;

  ComputeResiduals(view_graph, images, all_rig_pairs);
  VLOG(2) << "ComputeResiduals done";

  int iteration = 0;
  for (iteration = 0; iteration < options_.max_num_l1_iterations; iteration++) {
    VLOG(2) << "L1 ADMM iteration: " << iteration;

    last_norm = curr_norm;
    // use the current residual as b (Ax - b)

    tangent_space_step_.setZero();
    l1_solver.Solve(tangent_space_residual_, &tangent_space_step_);
    if (tangent_space_step_.array().isNaN().any()) {
      LOG(ERROR) << "nan error";
      iteration++;
      return false;
    }

    if (VLOG_IS_ON(2))
      LOG(INFO) << "residual:"
                << (sparse_matrix_ * tangent_space_step_ -
                    tangent_space_residual_)
                       .array()
                       .abs()
                       .sum();

    curr_norm = tangent_space_step_.norm();
    UpdateGlobalRotations(images, ref_image_ids);
    ComputeResiduals(view_graph, images, all_rig_pairs);

    // Check the residual. If it is small, stop
    // TODO: strange bug for the L1 solver: update norm state constant
    if (ComputeAverageStepSize(images, ref_image_ids) <
            options_.l1_step_convergence_threshold ||
        std::abs(last_norm - curr_norm) < EPS) {
      if (std::abs(last_norm - curr_norm) < EPS)
        LOG(INFO) << "std::abs(last_norm - curr_norm) < EPS";
      iteration++;
      break;
    }
    opt_l1_solver.max_num_iterations =
        std::min(opt_l1_solver.max_num_iterations * 2, 100);
  }
  VLOG(2) << "L1 ADMM total iteration: " << iteration;
  return true;
}

bool MultiRotationEstimator::SolveIRLS(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_set<image_t>& ref_image_ids,
    const std::vector<RefImagePairInfo>& all_rig_pairs) {
  // TODO: Determine what is the best solver for this part
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> llt;

  // weight_matrix.setIdentity();
  // sparse_matrix_ = A_ori;

  llt.analyzePattern(sparse_matrix_.transpose() * sparse_matrix_);

  const double sigma = DegToRad(options_.irls_loss_parameter_sigma);
  VLOG(2) << "sigma: " << options_.irls_loss_parameter_sigma;

  Eigen::ArrayXd weights_irls(sparse_matrix_.rows());
  Eigen::SparseMatrix<double> at_weight;

  weights_irls.segment(sparse_matrix_.rows() - 3, 3).setConstant(1);

  ComputeResiduals(view_graph, images, all_rig_pairs);
  int iteration = 0;
  for (iteration = 0; iteration < options_.max_num_irls_iterations;
       iteration++) {
    VLOG(2) << "IRLS iteration: " << iteration;

    // Compute the weights for IRLS
    int curr_pos = 0;
    for (const auto& ref_image_pair : all_rig_pairs) {
      double err_squared = 0;
      double w = 0;
      err_squared = tangent_space_residual_.segment(curr_pos, 3).squaredNorm();

      // Compute the weight
      if (options_.weight_type ==
          MultiRotationEstimatorOptions::GEMAN_MCCLURE) {
        double tmp = err_squared + sigma * sigma;
        w = sigma * sigma / (tmp * tmp);
      } else if (options_.weight_type ==
                 MultiRotationEstimatorOptions::HALF_NORM) {
        w = std::min(std::pow(err_squared, (0.5 - 2) / 2), 1e8);
      }

      if (std::isnan(w)) {
        LOG(ERROR) << "nan weight!";
        return false;
      }

      weights_irls.segment(curr_pos, 3).setConstant(w);
      curr_pos += 3;
    }

    // Update the factorization for the weighted values.
    at_weight = sparse_matrix_.transpose() * weights_irls.matrix().asDiagonal();

    llt.factorize(at_weight * sparse_matrix_);

    // Solve the least squares problem..
    tangent_space_step_.setZero();
    tangent_space_step_ = llt.solve(at_weight * tangent_space_residual_);
    UpdateGlobalRotations(images, ref_image_ids);
    ComputeResiduals(view_graph, images, all_rig_pairs);

    // Check the residual. If it is small, stop
    if (ComputeAverageStepSize(images, ref_image_ids) <
        options_.irls_step_convergence_threshold) {
      iteration++;
      break;
    }
  }
  VLOG(2) << "IRLS total iteration: " << iteration;

  return true;
}

void MultiRotationEstimator::UpdateGlobalRotations(
    std::unordered_map<image_t, Image>& images,
    const std::unordered_set<image_t>& ref_image_ids) {
  for (const auto& image_id : ref_image_ids) {
    const auto& image = images.at(image_id);
    if (!image.is_registered) continue;

    image_t vector_idx = image_id_to_idx_[image_id];

    Eigen::Matrix3d R_ori =
        AngleAxisToRotation(rotation_estimated_.segment(vector_idx, 3));

    rotation_estimated_.segment(vector_idx, 3) = RotationToAngleAxis(
        R_ori *
        AngleAxisToRotation(-tangent_space_step_.segment(vector_idx, 3)));
  }
}

void MultiRotationEstimator::ComputeResiduals(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::vector<RefImagePairInfo>& all_rig_pairs) {
  int curr_pos = 0;
  for (const auto& ref_image_pair : all_rig_pairs) {
    const auto& image_pair = view_graph.image_pairs.at(ref_image_pair.pair_id);
    image_t image_id1 = ref_image_pair.ref_image_id1;
    image_t image_id2 = ref_image_pair.ref_image_id2;

    image_t idx1 = image_id_to_idx_[image_id1];
    image_t idx2 = image_id_to_idx_[image_id2];

    Eigen::Matrix3d R_1, R_2;

    R_1 = AngleAxisToRotation(
        rotation_estimated_.segment(image_id_to_idx_[image_id1], 3));

    R_2 = AngleAxisToRotation(
        rotation_estimated_.segment(image_id_to_idx_[image_id2], 3));

    tangent_space_residual_.segment(curr_pos, 3) = -RotationToAngleAxis(
        R_2.transpose() * ref_image_pair.ref_rel_rotation.toRotationMatrix() *
        R_1);

    curr_pos += 3;
  }

  tangent_space_residual_.segment(tangent_space_residual_.size() - 3, 3) =
      RotationToAngleAxis(
          AngleAxisToRotation(fixed_camera_rotation_).transpose() *
          AngleAxisToRotation(rotation_estimated_.segment(
              image_id_to_idx_[fixed_camera_id_], 3)));
}

double MultiRotationEstimator::ComputeAverageStepSize(
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_set<image_t>& ref_image_ids) {
  double total_update = 0;
  for (const auto& image_id : ref_image_ids) {
    const auto& image = images.at(image_id);
    if (!image.is_registered) continue;

    total_update +=
        tangent_space_step_.segment(image_id_to_idx_[image_id], 3).norm();
  }
  return total_update / image_id_to_idx_.size();
}

}  // namespace MGSfM