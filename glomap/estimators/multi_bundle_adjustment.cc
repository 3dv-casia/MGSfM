#include "multi_bundle_adjustment.h"

#include <colmap/estimators/cost_functions.h>
#include <colmap/estimators/manifold.h>
#include <colmap/sensor/models.h>
#include <colmap/util/cuda.h>
#include <colmap/util/misc.h>

namespace MGSfM {
using namespace glomap;

template <typename CameraModel>
class MultiReprojErrorCostFunction {
 public:
  explicit MultiReprojErrorCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            MultiReprojErrorCostFunction<CameraModel>,
            2,
            4,
            3,
            4,
            3,
            3,
            CameraModel::num_params>(
        new MultiReprojErrorCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const rel_from_ref_rotation,
                  const T* const rel_from_ref_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        colmap::EigenQuaternionMap<T>(rel_from_ref_rotation) *
            (colmap::EigenQuaternionMap<T>(cam_from_world_rotation) *
                 colmap::EigenVector3Map<T>(point3D) +
             colmap::EigenVector3Map<T>(cam_from_world_translation)) +
        colmap::EigenVector3Map<T>(rel_from_ref_translation);
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

bool MultiBundleAdjuster::Solve(
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, Rigid3d>& rel_from_refs,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  // Check if the input data is valid
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (tracks.empty()) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }

  // Reset the problem
  Reset();

  // Add the constraints that the point tracks impose on the problem
  AddPointToCameraConstraints(
      view_graph, cameras, images, tracks, rel_from_refs, image_ref_id);

  // Add the cameras and points to the parameter groups for schur-based
  // optimization
  AddCamerasAndPointsToParameterGroups(cameras, images, tracks, rel_from_refs);

  // Parameterize the variables
  ParameterizeVariables(cameras, images, tracks, rel_from_refs);

  // Set the solver options.
  ceres::Solver::Summary summary;

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

  // Do not use the iterative solver, as it does not seem to be helpful
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;

  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  if (VLOG_IS_ON(2))
    LOG(INFO) << summary.FullReport();
  else
    LOG(INFO) << summary.BriefReport();

  for (auto& [image_id, image] : images) {
    if (image.camera_id == 1) continue;
    image_t ref_image_id = image_ref_id.at(image_id);
    auto& rel_from_ref = rel_from_refs.at(image.camera_id);
    image.cam_from_world.rotation =
        rel_from_ref.rotation * images.at(ref_image_id).cam_from_world.rotation;
    image.cam_from_world.translation =
        rel_from_ref.rotation *
            images.at(ref_image_id).cam_from_world.translation +
        rel_from_ref.translation;
  }

  for (auto& [image_id, image] : images) {
    if (image.camera_id != 1) continue;
    image_t ref_image_id = image_ref_id.at(image_id);
    auto& rel_from_ref = rel_from_refs.at(image.camera_id);
    image.cam_from_world.rotation =
        rel_from_ref.rotation * images.at(ref_image_id).cam_from_world.rotation;
    image.cam_from_world.translation =
        rel_from_ref.rotation *
            images.at(ref_image_id).cam_from_world.translation +
        rel_from_ref.translation;
  }
  return summary.IsSolutionUsable();
}

void MultiBundleAdjuster::Reset() {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();
}

void MultiBundleAdjuster::AddPointToCameraConstraints(
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, Rigid3d>& rel_from_refs,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;

    std::vector<Observation> observation_new;
    observation_new.reserve(track.observations.size());
    for (const auto& observation : track.observations) {
      auto it = images.find(observation.first);
      if (it == images.end()) continue;
      const Image& image = it->second;
      if (!image.is_registered || (image.cam_from_world * track.xyz)(2) < 1e-3)
        continue;
      observation_new.emplace_back(observation);
    }
    if (observation_new.size() != track.observations.size()) {
      track.observations = observation_new;
    }
    if (track.observations.size() < options_.min_num_view_per_track) continue;

    for (const auto& [image_id, feature_id] : track.observations) {
      Image& image = images.at(image_id);

      ceres::CostFunction* cost_function =
          colmap::CreateCameraCostFunction<MultiReprojErrorCostFunction>(
              cameras[image.camera_id].model_id, image.features[feature_id]);

      image_t ref_image_id = image_ref_id.at(image_id);

      if (cost_function != nullptr) {
        problem_->AddResidualBlock(
            cost_function,
            loss_function_.get(),
            images.at(ref_image_id).cam_from_world.rotation.coeffs().data(),
            images.at(ref_image_id).cam_from_world.translation.data(),
            rel_from_refs.at(image.camera_id).rotation.coeffs().data(),
            rel_from_refs.at(image.camera_id).translation.data(),
            tracks[track_id].xyz.data(),
            cameras[image.camera_id].params.data());
      } else {
        LOG(ERROR) << "Camera model not supported: "
                   << colmap::CameraModelIdToName(
                          cameras[image.camera_id].model_id);
      }
    }
  }
}

void MultiBundleAdjuster::AddCamerasAndPointsToParameterGroups(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, Rigid3d>& rel_from_refs) {
  if (tracks.size() == 0) return;

  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();
  // Add point parameters to group 0.
  for (auto& [track_id, track] : tracks) {
    if (problem_->HasParameterBlock(track.xyz.data()))
      parameter_ordering->AddElementToGroup(track.xyz.data(), 0);
  }

  // Add camera parameters to group 1.
  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(image.cam_from_world.translation.data())) {
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.translation.data(), 1);
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.rotation.coeffs().data(), 1);
    }
  }

  for (auto& [camera_id, rel_from_ref] : rel_from_refs) {
    if (problem_->HasParameterBlock(rel_from_ref.translation.data())) {
      parameter_ordering->AddElementToGroup(rel_from_ref.translation.data(), 1);
      parameter_ordering->AddElementToGroup(
          rel_from_ref.rotation.coeffs().data(), 1);
    }
  }

  // Add camera parameters to group 1.
  for (auto& [camera_id, camera] : cameras) {
    if (problem_->HasParameterBlock(camera.params.data()))
      parameter_ordering->AddElementToGroup(camera.params.data(), 1);
  }
}

void MultiBundleAdjuster::ParameterizeVariables(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, Rigid3d>& rel_from_refs) {
  image_t center;

  // Parameterize rotations, and set rotations and translations to be constant
  int counter = 0;
  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(
            image.cam_from_world.rotation.coeffs().data())) {
      colmap::SetQuaternionManifold(
          problem_.get(), image.cam_from_world.rotation.coeffs().data());

      if (counter == 0) {
        center = image_id;
        counter++;
      }
      if (!options_.optimize_rotations)
        problem_->SetParameterBlockConstant(
            image.cam_from_world.rotation.coeffs().data());
      if (!options_.optimize_translation)
        problem_->SetParameterBlockConstant(
            image.cam_from_world.translation.data());
    }
  }

  for (auto& [camera_id, rel_from_ref] : rel_from_refs) {
    if (problem_->HasParameterBlock(rel_from_ref.rotation.coeffs().data())) {
      colmap::SetQuaternionManifold(problem_.get(),
                                    rel_from_ref.rotation.coeffs().data());
      if (!options_.optimize_rotations || !options_.optimize_internal_rotation)
        problem_->SetParameterBlockConstant(
            rel_from_ref.rotation.coeffs().data());
      if (!options_.optimize_translation ||
          !options_.optimize_internal_translation)
        problem_->SetParameterBlockConstant(rel_from_ref.translation.data());
    }
  }

  if (counter > 0) {
    // Set the first image to be fixed to remove the gauge ambiguity.
    problem_->SetParameterBlockConstant(
        images[center].cam_from_world.rotation.coeffs().data());
    problem_->SetParameterBlockConstant(
        images[center].cam_from_world.translation.data());
  }
  // Set camera 1 in rig to be fixed to remove the gauge ambiguity.
  problem_->SetParameterBlockConstant(
      rel_from_refs.at(1).rotation.coeffs().data());
  problem_->SetParameterBlockConstant(rel_from_refs.at(1).translation.data());

  // Parameterize the camera parameters, or set them to be constant if desired
  if (options_.optimize_intrinsics && !options_.optimize_principal_point) {
    for (auto& [camera_id, camera] : cameras) {
      if (problem_->HasParameterBlock(camera.params.data())) {
        std::vector<int> principal_point_idxs;
        for (auto idx : camera.PrincipalPointIdxs()) {
          principal_point_idxs.push_back(idx);
        }
        colmap::SetSubsetManifold(camera.params.size(),
                                  principal_point_idxs,
                                  problem_.get(),
                                  camera.params.data());
      }
    }

  } else if (!options_.optimize_intrinsics &&
             !options_.optimize_principal_point) {
    for (auto& [camera_id, camera] : cameras) {
      if (problem_->HasParameterBlock(camera.params.data())) {
        problem_->SetParameterBlockConstant(camera.params.data());
      }
    }
  }

  if (!options_.optimize_points) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
      }
    }
  }
}

}  // namespace MGSfM
