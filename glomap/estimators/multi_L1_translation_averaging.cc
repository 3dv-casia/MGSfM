#include "glomap/estimators/multi_L1_translation_averaging.h"

#include <colmap/util/threading.h>

#include <vector>

namespace MGSfM {
using namespace glomap;
int MultiL1TranslationAveraging::MultiSetupL1Solver(Eigen::VectorXd& b) {
  const int all_trans_size = pairs_indices_.size();

  const int cols = 3 * (image_id_to_index_.size() + rel_from_refs_.size() - 1) +
                   all_trans_size;
  const int rows = (3 + 1) * all_trans_size + 3;

  constraint_matrix_.resize(rows, cols);

  b.resize(rows);
  b.setZero();

  std::vector<Eigen::Triplet<double>> triplet_list(
      (6 + 3 + 18 + 1) * all_trans_size + 3);

  const int fix_index = image_id_to_index_.at(fix_id_);
  for (int i = 0; i < 3; i++)
    triplet_list[i] = Eigen::Triplet<double>(i, fix_index + i, 1.0);

  int row_base_index = 3;
  int triplet_base_index = 3;
  int geq_base_index = 3 * (all_trans_size + 1);
  int scale_index = cols - all_trans_size;

#pragma omp parallel for
  for (int i = 0; i < all_trans_size; i++) {
    const RefImagePairInfo& image_pair =
        ref_image_pairs_.at(pairs_indices_.at(i));

    const int ref1_index = image_id_to_index_.at(image_pair.ref_image_id1);
    const int ref2_index = image_id_to_index_.at(image_pair.ref_image_id2);

    const int row_index = row_base_index + 3 * i;
    const int geq_index = geq_base_index + i;
    int triplet_index = triplet_base_index + 28 * i;

    const auto& raw_image_pair = view_graph_.image_pairs.at(image_pair.pair_id);
    const auto& image1 = images_.at(raw_image_pair.image_id1);
    const auto& image2 = images_.at(raw_image_pair.image_id2);
    const auto inv_rotation1 =
        image1.cam_from_world.rotation.inverse().toRotationMatrix();
    const auto inv_rotation2 =
        image2.cam_from_world.rotation.inverse().toRotationMatrix();

    for (int k = 0; k < 3; k++) {
      triplet_list[triplet_index++] =
          Eigen::Triplet<double>(row_index + k, ref1_index + k, 1.0);
      triplet_list[triplet_index++] =
          Eigen::Triplet<double>(row_index + k, ref2_index + k, -1.0);
      triplet_list[triplet_index++] = Eigen::Triplet<double>(
          row_index + k, scale_index + i, -image_pair.rel_translation(k));
    }

    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < 3; j++) {
        if (image1.camera_id != 1)
          triplet_list[triplet_index++] =
              Eigen::Triplet<double>(row_index + k,
                                     3 * (image1.camera_id - 2) + j,
                                     -inv_rotation1(k, j));
        else
          triplet_list[triplet_index++] = Eigen::Triplet<double>(0, 0, 0);
        if (image2.camera_id != 1)
          triplet_list[triplet_index++] =
              Eigen::Triplet<double>(row_index + k,
                                     3 * (image2.camera_id - 2) + j,
                                     inv_rotation2(k, j));
        else
          triplet_list[triplet_index++] = Eigen::Triplet<double>(0, 0, 0);
      }
    }
    triplet_list[triplet_index++] =
        Eigen::Triplet<double>(geq_index, scale_index + i, 1.0);
  }

  constraint_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());

  b.tail(all_trans_size).setOnes();

  // Number of constraints
  return all_trans_size;
}

bool MultiL1TranslationAveraging::MultiEstimataTranslation(void) {
  LOG(INFO) << "Running multiple L1 translation averaging ...";
  image_id_to_index_.clear();
  image_id_to_index_.reserve(ref_image_ids_.size() + rel_from_refs_.size() - 1);
  int index = 3 * (rel_from_refs_.size() - 1);
  for (const auto& image_id : ref_image_ids_) {
    image_id_to_index_[image_id] = index;
    index += 3;
  }
  Eigen::VectorXd b;
  const int num_constraints = MultiSetupL1Solver(b);
  const int num_l1_residuals = constraint_matrix_.rows() - num_constraints;

  solution_.setZero(constraint_matrix_.cols());

  HETA::ConstrainedL1Solver::Options l1_options;
  l1_options.max_num_iterations = 2000;
  HETA::ConstrainedL1Solver solver(
      l1_options, constraint_matrix_, b, num_l1_residuals, num_constraints);

  solver.FastAdapSolve(&solution_);
  for (auto& [camera_id, rel_from_ref] : rel_from_refs_) {
    if (camera_id != 1)
      rel_from_ref.translation = solution_.segment<3>(3 * camera_id - 6);
    VLOG(2) << "Internal translation of camera " << camera_id << ": "
            << rel_from_ref.translation.transpose();
  }

  for (auto& [image_id, image] : images_) {
    if (!image.is_registered) continue;
    const auto& rel_from_ref = rel_from_refs_.at(image.camera_id);

    image.cam_from_world.translation =
        image.cam_from_world.rotation *
            (-solution_.segment<3>(
                image_id_to_index_.at(image_ref_id_.at(image_id)))) +
        rel_from_ref.translation;
  }
  return true;
}

}  // namespace MGSfM
