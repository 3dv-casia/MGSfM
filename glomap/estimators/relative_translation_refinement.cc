#include "glomap/estimators/relative_translation_refinement.h"

namespace HETA {
bool RelativeTranslationRefiner::SolveMatrix(
    Eigen::MatrixXd& constraint_matrix,
    Eigen::Vector3d& relative_translation,
    Eigen::Array<bool, Eigen::Dynamic, 1>& masks) {
  const double eps = 1e-5;
  const int IRLS_max_iterations = 100;
  const int IRLS_max_inner_iterations = 10;
  const double square_residual_threshold =
      options_.residual_threshold * options_.residual_threshold;

  Eigen::VectorXd weights(constraint_matrix.cols());

  if (options_.use_prior_relative_translation) {
    weights =
        (relative_translation.transpose() * constraint_matrix).array().abs();
  } else {
    weights.setConstant(1.0);
  }

  // Solve for the relative positions using a robust IRLS.
  double cost = weights.sum();
  // double cost = 0;
  int num_inner_iterations = 0;
  int i;
  Eigen::Matrix3d lhs;
  for (i = 0; i < IRLS_max_iterations &&
              num_inner_iterations < IRLS_max_inner_iterations;
       i++) {
    // Apply the weights to the constraint matrix.
    lhs = (constraint_matrix * weights.asDiagonal() *
           constraint_matrix.transpose())
              .selfadjointView<Eigen::Lower>();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(lhs);
    CHECK_EQ(eigensolver.info(), Eigen::Success);
    relative_translation = eigensolver.eigenvectors().col(0);

    // Update the weights based on the current errors.
    weights =
        (relative_translation.transpose() * constraint_matrix).array().abs();

    // Compute the new cost.
    const double new_cost = weights.sum();

    // Check for convergence.
    const double delta = std::max(std::abs(cost - new_cost),
                                  1 - relative_translation.squaredNorm());

    // If we have good convergence, attempt an inner iteration.
    if (delta <= eps) {
      ++num_inner_iterations;
    } else {
      num_inner_iterations = 0;
    }

    cost = new_cost;

    masks = weights.array() < options_.residual_threshold;
    // GEMAN-MCCLURE kernel
    weights = square_residual_threshold /
              (weights.array().square() + square_residual_threshold).square();
  }

  return i < IRLS_max_iterations;
}

void RelativeTranslationRefiner::SelectFeatureMatches() {
  std::vector<glomap::image_pair_t> valid_pair_ids;
  for (auto& [image_pair_id, image_pair] : view_graph_.image_pairs) {
    if (!image_pair.is_valid) continue;
    valid_pair_ids.emplace_back(image_pair_id);
  }

  const size_t kNumChunks = 10;
  size_t inverval = std::ceil(valid_pair_ids.size() / kNumChunks);
  LOG(INFO) << "Refining relative translation for " << valid_pair_ids.size()
            << " pairs";
  for (size_t chunk_id = 0; chunk_id < kNumChunks; chunk_id++) {
    std::cout << "\r Refining relative translation: " << chunk_id * kNumChunks
              << "%" << std::flush;
    const size_t start = chunk_id * inverval;
    const size_t end =
        std::min((chunk_id + 1) * inverval, valid_pair_ids.size());
#pragma omp parallel for schedule(dynamic)
    for (size_t pair_idx = start; pair_idx < end; pair_idx++) {
      glomap::ImagePair& image_pair =
          view_graph_.image_pairs.at(valid_pair_ids.at(pair_idx));
      const glomap::Image& image1 = images_.at(image_pair.image_id1);
      const glomap::Image& image2 = images_.at(image_pair.image_id2);

      const uint32_t raw_inlier_size = image_pair.inliers.size();

      std::vector<int> tri_angle_valid_indices;
      tri_angle_valid_indices.reserve(raw_inlier_size);
      Eigen::MatrixXd constraint_matrix(3, raw_inlier_size);

      const auto relative_rotation = image2.cam_from_world.rotation *
                                     image1.cam_from_world.rotation.inverse();

      for (const auto& row_index : image_pair.inliers) {
        const auto& point_index = image_pair.matches.row(row_index);
        const Eigen::Vector3d& feature_undist1 =
            image1.features_undist.at(point_index(0));
        const Eigen::Vector3d& feature_undist2 =
            image2.features_undist.at(point_index(1));
        if (feature_undist1.array().isNaN().any() ||
            feature_undist2.array().isNaN().any()) {
          continue;
        }

        const Eigen::Vector3d cross_feature_ray =
            image2.features_undist.at(point_index(1))
                .cross(relative_rotation *
                       image1.features_undist.at(point_index(0)));
        const double tri_angle = cross_feature_ray.norm();
        if (tri_angle > options_.lower_sin_angle_threshold &&
            tri_angle < options_.upper_sin_angle_threshold) {
          constraint_matrix.col(tri_angle_valid_indices.size()) =
              cross_feature_ray;
          tri_angle_valid_indices.emplace_back(row_index);
        }
      }

      const uint32_t tri_angle_valid_size = tri_angle_valid_indices.size();
      // If the number of inliers is insufficient or the ratio of low-parallax
      // feature matches is too high, we discard this translation.
      if (tri_angle_valid_size <
          std::max(options_.min_feature_correspondence_num,
                   static_cast<uint32_t>(raw_inlier_size *
                                         options_.min_valid_parallax_ratio))) {
        image_pair.is_valid = false;
        continue;
      }
      constraint_matrix.conservativeResize(3, tri_angle_valid_size);
      // Refine relative translations with IRLS
      Eigen::Vector3d relative_translation =
          image_pair.cam2_from_cam1.translation.normalized();

      Eigen::Array<bool, Eigen::Dynamic, 1> solve_masks(tri_angle_valid_size);
      if (!SolveMatrix(constraint_matrix, relative_translation, solve_masks)) {
        image_pair.is_valid = false;
        continue;
      }
      const int min_inlier_threshold = static_cast<int>(
          std::max(options_.min_feature_correspondence_num,
                   static_cast<uint32_t>(options_.min_IRLS_inlier_ratio *
                                         tri_angle_valid_size)));
      const int solve_valid_size = solve_masks.count();
      if (solve_valid_size < min_inlier_threshold) {
        image_pair.is_valid = false;
        continue;
      }
      std::vector<int> solve_valid_indices;
      solve_valid_indices.reserve(solve_valid_size);
      for (int i = 0; i < tri_angle_valid_size; i++) {
        if (solve_masks(i)) {
          solve_valid_indices.emplace_back(tri_angle_valid_indices[i]);
        }
      }
      CHECK_EQ(solve_valid_indices.size(), solve_valid_size);

      // Check Cheirality
      Eigen::Matrix<bool, Eigen::Dynamic, 1> front_masks(solve_valid_size);
      for (int i = 0; i < solve_valid_size; i++) {
        const auto& point_index =
            image_pair.matches.row(solve_valid_indices.at(i));
        const Eigen::Vector3d feature_ray_1 =
            relative_rotation * image1.features_undist.at(point_index(0));
        const Eigen::Vector3d& feature_ray_2 =
            image2.features_undist.at(point_index(1));

        const double dir1_dir2 = feature_ray_1.dot(feature_ray_2);
        const double dir1_pos = feature_ray_1.dot(relative_translation);
        const double dir2_pos = feature_ray_2.dot(relative_translation);

        front_masks(i) = ((dir1_pos - dir1_dir2 * dir2_pos < 0 &&
                           dir1_dir2 * dir1_pos - dir2_pos < 0) &&
                          dir1_pos * dir2_pos - dir1_dir2 < 0);
      }
      int front_count = front_masks.count();
      // Judge the direction of translation
      bool reverse_flag = front_count < (solve_valid_size >> 1);
      if (reverse_flag) {
        front_count = solve_valid_size - front_count;
        relative_translation = -relative_translation;
      }
      if (front_count < options_.min_feature_correspondence_num) {
        image_pair.is_valid = false;
        continue;
      }
      // return inliers and the re-estimated translation
      std::vector<int> final_inlier_indices;
      final_inlier_indices.reserve(front_count);
      for (int i = 0; i < solve_valid_size; i++) {
        if (front_masks(i) != reverse_flag) {
          final_inlier_indices.emplace_back(solve_valid_indices[i]);
        }
      }
      CHECK_EQ(final_inlier_indices.size(), front_count);
      std::swap(final_inlier_indices, image_pair.inliers);
      image_pair.cam2_from_cam1.translation = relative_translation;
    }
  }
  std::cout << "\r Refining relative translation: 100%" << std::endl;
}
}  // namespace HETA