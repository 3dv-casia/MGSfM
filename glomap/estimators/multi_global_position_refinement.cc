#include "glomap/estimators/multi_global_position_refinement.h"

#include <colmap/estimators/cost_functions.h>
#include <colmap/estimators/manifold.h>
#include <colmap/util/cuda.h>
#include <colmap/util/misc.h>
#include <colmap/util/types.h>

namespace MGSfM {
using namespace glomap;
MultiPositionRefiner::MultiPositionRefiner(
    const MultiPositionRefinerOptions& options)
    : options_(options) {}

bool MultiPositionRefiner::Solve(
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (view_graph.image_pairs.empty() &&
      options_.constraint_type != MultiPositionRefinerOptions::ONLY_POINTS) {
    LOG(ERROR) << "Number of image_pairs = " << view_graph.image_pairs.size();
    return false;
  }
  if (tracks.empty() &&
      options_.constraint_type != MultiPositionRefinerOptions::ONLY_CAMERAS) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }
  if (options_.constraint_type != MultiPositionRefinerOptions::ONLY_POINTS)
    LOG(INFO) << "Setting up the Camera-to-Camera Angle Refinement Problem";
  if (options_.constraint_type != MultiPositionRefinerOptions::ONLY_CAMERAS)
    LOG(INFO) << "Setting up the Camera-to-Point Angle Refinement Problem";

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction(options_.angle_thresh);

  for (auto& [image_id, image] : images) {
    image.cam_from_world.translation = image.Center();
  }

  // Add the camera to camera constraints to the problem.
  if (options_.constraint_type != MultiPositionRefinerOptions::ONLY_POINTS) {
    AddCameraToCameraConstraints(
        view_graph, images, rel_from_refs, image_ref_id);
  }

  // Add the point to camera constraints to the problem.
  if (options_.constraint_type != MultiPositionRefinerOptions::ONLY_CAMERAS) {
    AddPointToCameraConstraints(
        cameras, images, tracks, rel_from_refs, image_ref_id);
  }

  if (options_.regularize_trajectory) {
    AddTrajectoryRegularizationConstraints(view_graph, images, image_ref_id);
  }

  AddCamerasAndPointsToParameterGroups(images, tracks, rel_from_refs);

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(images, tracks, rel_from_refs);

  LOG(INFO) << "Solving the Angle Refinement Problem";

  ceres::Solver::Summary summary;
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  for (auto& [image_id, image] : images) {
    if (!image.is_registered || image.camera_id == 1) continue;
    image_t ref_image_id = image_ref_id.at(image_id);
    image.cam_from_world.translation =
        images.at(ref_image_id).cam_from_world.translation +
        image.cam_from_world.rotation.inverse() *
            (-rel_from_refs.at(image.camera_id).translation);
  }
  for (auto& [image_id, image] : images) {
    if (!image.is_registered || image.camera_id != 1) continue;
    image_t ref_image_id = image_ref_id.at(image_id);
    image.cam_from_world.translation =
        images.at(ref_image_id).cam_from_world.translation +
        image.cam_from_world.rotation.inverse() *
            (-rel_from_refs.at(image.camera_id).translation);
  }

  ConvertResults(images);
  return summary.IsSolutionUsable();
}

void MultiPositionRefiner::AddCameraToCameraConstraints(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    image_t ref_image_id1 = image_ref_id.at(image_pair.image_id1);
    image_t ref_image_id2 = image_ref_id.at(image_pair.image_id2);
    camera_t camera_id1 = images.at(image_pair.image_id1).camera_id;
    camera_t camera_id2 = images.at(image_pair.image_id2).camera_id;

    Eigen::Vector3d translation_obs =
        -(images[image_pair.image_id2].cam_from_world.rotation.inverse() *
          image_pair.cam2_from_cam1.translation)
             .normalized();

    ceres::CostFunction* cost_function;

    if (ref_image_id1 == ref_image_id2) {
      cost_function = IC2CAngleError::Create(
          translation_obs,
          images.at(image_pair.image_id1).cam_from_world.rotation.inverse(),
          images.at(image_pair.image_id2).cam_from_world.rotation.inverse());
      problem_->AddResidualBlock(
          cost_function,
          loss_function_.get(),
          rel_from_refs.at(camera_id1).translation.data(),
          rel_from_refs.at(camera_id2).translation.data());
    } else if (camera_id1 == camera_id2) {
      cost_function = ESC2CAngleError::Create(
          translation_obs,
          images.at(image_pair.image_id1).cam_from_world.rotation.inverse(),
          images.at(image_pair.image_id2).cam_from_world.rotation.inverse());
      problem_->AddResidualBlock(
          cost_function,
          loss_function_.get(),
          images[ref_image_id1].cam_from_world.translation.data(),
          images[ref_image_id2].cam_from_world.translation.data(),
          rel_from_refs.at(camera_id1).translation.data());
    } else {
      cost_function = EMC2CAngleError::Create(
          translation_obs,
          images.at(image_pair.image_id1).cam_from_world.rotation.inverse(),
          images.at(image_pair.image_id2).cam_from_world.rotation.inverse());
      problem_->AddResidualBlock(
          cost_function,
          loss_function_.get(),
          images[ref_image_id1].cam_from_world.translation.data(),
          images[ref_image_id2].cam_from_world.translation.data(),
          rel_from_refs.at(camera_id1).translation.data(),
          rel_from_refs.at(camera_id2).translation.data());
    }
  }

  VLOG(2) << problem_->NumResidualBlocks()
          << " camera to camera constraints were added to the position "
             "estimation problem.";
}
void MultiPositionRefiner::AddTrajectoryRegularizationConstraints(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  std::unordered_set<image_t> ref_images;
  std::unordered_set<std::pair<image_t, image_t>> ref_pairs;
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    // if (!image_pair.is_valid) continue;
    image_t ref_image_id1 = image_ref_id.at(image_pair.image_id1);
    image_t ref_image_id2 = image_ref_id.at(image_pair.image_id2);

    if (ref_image_id1 == ref_image_id2) continue;
    if (ref_image_id1 > ref_image_id2) std::swap(ref_image_id1, ref_image_id2);
    if (ref_image_id2 - ref_image_id1 > 3) continue;
    ref_pairs.emplace(ref_image_id1, ref_image_id2);
    if (images.at(ref_image_id1).is_registered)
      ref_images.emplace(ref_image_id1);
    if (images.at(ref_image_id2).is_registered)
      ref_images.emplace(ref_image_id2);
  }
  ceres::CostFunction* reg_cost_function =
      TrajectoryRegularization::Create(options_.regularize_weight);
  for (const auto& ref_image_id : ref_images) {
    if (!(ref_images.count(ref_image_id - 1) &&
          ref_images.count(ref_image_id + 1)))
      continue;
    int all_count = ref_pairs.count({ref_image_id - 1, ref_image_id}) +
                    ref_pairs.count({ref_image_id, ref_image_id + 1}) +
                    ref_pairs.count({ref_image_id - 1, ref_image_id + 1});
    if (all_count < 2) continue;
    problem_->AddResidualBlock(
        reg_cost_function,
        nullptr,
        images[ref_image_id - 1].cam_from_world.translation.data(),
        images[ref_image_id].cam_from_world.translation.data(),
        images[ref_image_id + 1].cam_from_world.translation.data());
  }
}

void MultiPositionRefiner::AddPointToCameraConstraints(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  // The number of camera-to-camera constraints coming from the relative poses

  const size_t num_cam_to_cam = problem_->NumResidualBlocks();
  // Find the tracks that are relevant to the current set of cameras
  const size_t num_pt_to_cam = tracks.size();

  VLOG(2) << num_pt_to_cam
          << " point to camera constriants were added to the position "
             "estimation problem.";

  if (num_pt_to_cam == 0) return;

  if (loss_function_ptcam_uncalibrated_ == nullptr) {
    loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
        loss_function_.get(), 0.5, ceres::DO_NOT_TAKE_OWNERSHIP);
  }

  loss_function_ptcam_calibrated_ = loss_function_;

  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;
    std::vector<Observation> observation_new;
    observation_new.reserve(track.observations.size());
    for (const auto& observation : track.observations) {
      auto it = images.find(observation.first);
      if (it == images.end()) continue;
      const Image& image = it->second;
      if (!image.is_registered) continue;
      Eigen::Vector3d pt_calc = image.cam_from_world.rotation *
                                (track.xyz - image.cam_from_world.translation);
      if (pt_calc(2) < 1e-3) continue;
      if (image.features_undist[observation.second].array().isNaN().any())
        continue;
      observation_new.emplace_back(observation);
    }
    if (observation_new.size() != track.observations.size()) {
      track.observations = observation_new;
    }
    if (track.observations.size() < options_.min_num_view_per_track) continue;
    AddTrackToProblem(cameras, images, track, rel_from_refs, image_ref_id);
  }
}

void MultiPositionRefiner::AddTrackToProblem(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    Track& track,
    std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  // For each view in the track add the point to camera correspondences.
  for (const auto& [image_id, feature_id] : track.observations) {
    Image& image = images.at(image_id);
    image_t ref_image_id = image_ref_id.at(image_id);
    camera_t camera_id = image.camera_id;

    const Eigen::Vector3d translation_obs =
        (image.cam_from_world.rotation.inverse() *
         image.features_undist.at(feature_id))
            .normalized();

    ceres::CostFunction* cost_function = C2PMulti1DSfMError::Create(
        translation_obs, image.cam_from_world.rotation.inverse());

    // For calibrated and uncalibrated cameras, use different loss functions
    // Down weight the uncalibrated cameras
    (cameras[camera_id].has_prior_focal_length)
        ? problem_->AddResidualBlock(
              cost_function,
              loss_function_ptcam_calibrated_.get(),
              images.at(ref_image_id).cam_from_world.translation.data(),
              track.xyz.data(),
              rel_from_refs.at(camera_id).translation.data())
        : problem_->AddResidualBlock(
              cost_function,
              loss_function_ptcam_uncalibrated_.get(),
              images.at(ref_image_id).cam_from_world.translation.data(),
              track.xyz.data(),
              rel_from_refs.at(camera_id).translation.data());
  }
}

void MultiPositionRefiner::AddCamerasAndPointsToParameterGroups(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  int group_id = 0;

  if (tracks.size() > 0 &&
      options_.constraint_type != MultiPositionRefinerOptions::ONLY_CAMERAS) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data()))
        parameter_ordering->AddElementToGroup(track.xyz.data(), group_id);
    }
    group_id++;
  }

  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(image.cam_from_world.translation.data())) {
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.translation.data(), group_id);
    }
  }

  for (auto& [camera_id, rel_from_ref] : rel_from_refs) {
    if (problem_->HasParameterBlock(rel_from_ref.translation.data())) {
      parameter_ordering->AddElementToGroup(rel_from_ref.translation.data(),
                                            group_id);
    }
  }
}

void MultiPositionRefiner::ParameterizeVariables(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs) {
  // For the global positioning, do not set any camera to be constant for
  // easier convergence

  // If do not optimize the positions, set the camera positions to be constant
  if (!options_.optimize_positions) {
    for (auto& [image_id, image] : images)
      if (problem_->HasParameterBlock(image.cam_from_world.translation.data()))
        problem_->SetParameterBlockConstant(
            image.cam_from_world.translation.data());
  }

  // If do not optimize the 3D points, set the 3D points to be constant
  if (!options_.optimize_points &&
      options_.constraint_type != MultiPositionRefinerOptions::ONLY_CAMERAS) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
      }
    }
  }

  // fix the translation of camera 1 in rig to be constant
  if (problem_->HasParameterBlock(rel_from_refs.at(1).translation.data()))
    problem_->SetParameterBlockConstant(rel_from_refs.at(1).translation.data());

  if (!options_.optimize_positions || !options_.optimize_internal_translation) {
    for (auto& [camera_id, rel_from_ref] : rel_from_refs) {
      if (problem_->HasParameterBlock(rel_from_ref.translation.data())) {
        problem_->SetParameterBlockConstant(rel_from_ref.translation.data());
      }
    }
  }

  int num_images = images.size();
#ifdef GLOMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu && num_images >= options_.min_num_images_gpu_solver) {
    cuda_solver_enabled = true;
    options_.solver_options.dense_linear_algebra_library_type = ceres::CUDA;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (options_.use_gpu && num_images >= options_.min_num_images_gpu_solver) {
    cuda_solver_enabled = true;
    options_.solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices =
        colmap::CSVToVector<int>(options_.gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    colmap::SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // GLOMAP_CUDA_ENABLED

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  if (options_.constraint_type != MultiPositionRefinerOptions::ONLY_CAMERAS) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

void MultiPositionRefiner::ConvertResults(
    std::unordered_map<image_t, Image>& images) {
  // translation now stores the camera position, needs to convert back to
  // translation
  for (auto& [image_id, image] : images) {
    image.cam_from_world.translation =
        -(image.cam_from_world.rotation * image.cam_from_world.translation);
  }
}

}  // namespace MGSfM