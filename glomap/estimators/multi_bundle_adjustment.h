#pragma once

#include "glomap/estimators/optimization_base.h"
#include "glomap/scene/types_sfm.h"
#include "glomap/types.h"

#include <ceres/ceres.h>

namespace MGSfM {
using namespace glomap;

struct MultiBundleAdjusterOptions : public OptimizationBaseOptions {
 public:
  // Flags for which parameters to optimize
  bool optimize_rotations = true;
  bool optimize_translation = true;
  bool optimize_intrinsics = true;
  bool optimize_principal_point = false;
  bool optimize_points = true;
  bool optimize_internal_translation = true;
  bool optimize_internal_rotation = true;

  bool use_gpu = true;
  std::string gpu_index = "-1";
  int min_num_images_gpu_solver = 50;

  // Constrain the minimum number of views per track
  int min_num_view_per_track = 3;

  MultiBundleAdjusterOptions() : OptimizationBaseOptions() {
    thres_loss_function = 1.;
    solver_options.max_num_iterations = 200;
  }
  std::shared_ptr<ceres::LossFunction> CreateLossFunction() {
    return std::make_shared<ceres::HuberLoss>(thres_loss_function);
  }
};

class MultiBundleAdjuster {
 public:
  MultiBundleAdjuster(const MultiBundleAdjusterOptions& options)
      : options_(options) {}

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  bool Solve(const ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks,
             std::unordered_map<camera_t, Rigid3d>& rel_from_refs,
             const std::unordered_map<image_t, image_t>& image_ref_id);

  MultiBundleAdjusterOptions& GetOptions() { return options_; }

 private:
  // Reset the problem
  void Reset();

  // Add tracks to the problem
  void AddPointToCameraConstraints(
      const ViewGraph& view_graph,
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      std::unordered_map<camera_t, Rigid3d>& rel_from_refs,
      const std::unordered_map<image_t, image_t>& image_ref_id);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      std::unordered_map<camera_t, Rigid3d>& rel_from_refs);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      std::unordered_map<camera_t, Rigid3d>& rel_from_refs);

  MultiBundleAdjusterOptions options_;

  std::unique_ptr<ceres::Problem> problem_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
};

}  // namespace MGSfM
