#pragma once

#include "glomap/estimators/cost_function.h"
#include "glomap/estimators/optimization_base.h"
#include "glomap/math/rigid3d.h"
#include "glomap/scene/types_sfm.h"
#include "glomap/types.h"

namespace MGSfM {
using namespace glomap;

struct MultiPositionRefinerOptions : public OptimizationBaseOptions {
  // ONLY_POINTS is recommended
  enum ConstraintType {
    // only include camera to point constraints
    ONLY_POINTS,
    // only include camera to camera constraints
    ONLY_CAMERAS,
    // treat each contribution from camera to point and camera to camera equally
    POINTS_AND_CAMERAS,
  };

  // Flags for which parameters to optimize
  bool optimize_positions = true;
  bool optimize_points = true;
  bool optimize_internal_translation = true;
  bool regularize_trajectory = true;

  double regularize_weight = 1e-2;
  double angle_thresh = 0.5;

  bool use_gpu = true;
  std::string gpu_index = "-1";
  int min_num_images_gpu_solver = 50;

  // Constrain the minimum number of views per track
  int min_num_view_per_track = 3;

  // the type of global positioning
  ConstraintType constraint_type = ONLY_POINTS;

  MultiPositionRefinerOptions() : OptimizationBaseOptions() {
    solver_options.max_num_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction(double angle_thresh) {
    return std::make_shared<ceres::CauchyLoss>(
        std::sin(DegToRad(angle_thresh)));
  }
};

class MultiPositionRefiner {
 public:
  MultiPositionRefiner(const MultiPositionRefinerOptions& options);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  bool Solve(const ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks,
             std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs,
             const std::unordered_map<image_t, image_t>& image_ref_id);

  // During the optimization, the camera translation is set to be the camera
  // center Convert the results back to camera poses
  void ConvertResults(std::unordered_map<image_t, Image>& images);

  MultiPositionRefinerOptions& GetOptions() { return options_; }

 protected:
  // Creates camera to camera constraints from relative translations. (3D)
  void AddCameraToCameraConstraints(
      const ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs,
      const std::unordered_map<image_t, image_t>& image_ref_id);

  void AddTrajectoryRegularizationConstraints(
      const ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      const std::unordered_map<image_t, image_t>& image_ref_id);

  // Add tracks to the problem
  void AddPointToCameraConstraints(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs,
      const std::unordered_map<image_t, image_t>& image_ref_id);

  // Add a single track to the problem
  void AddTrackToProblem(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      Track& track,
      std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs,
      const std::unordered_map<image_t, image_t>& image_ref_id);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      std::unordered_map<camera_t, colmap::Rigid3d>& rel_from_refs);

  // Data members
  MultiPositionRefinerOptions options_;

  std::unique_ptr<ceres::Problem> problem_;

  // loss functions for reweighted terms
  std::shared_ptr<ceres::LossFunction> loss_function_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_uncalibrated_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_calibrated_;
};

}  // namespace MGSfM
