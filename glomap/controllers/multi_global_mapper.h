#pragma once

#include "glomap/controllers/track_establishment.h"
#include "glomap/controllers/track_retriangulation.h"
#include "glomap/estimators/global_rotation_averaging.h"
#include "glomap/estimators/multi_bundle_adjustment.h"
#include "glomap/estimators/multi_global_position_refinement.h"
#include "glomap/estimators/multi_global_rotation_averaging.h"
#include "glomap/estimators/relative_translation_refinement.h"
#include "glomap/estimators/relpose_estimation.h"
#include "glomap/estimators/view_graph_calibration.h"
#include "glomap/types.h"

#include <colmap/scene/database.h>

namespace MGSfM {
using namespace glomap;
struct MultiGlobalMapperOptions {
  // Options for each component
  ViewGraphCalibratorOptions opt_vgcalib;
  RelativePoseEstimationOptions opt_relpose;
  RotationEstimatorOptions opt_ra;
  MultiRotationEstimatorOptions opt_mre;
  TrackEstablishmentOptions opt_track;
  HETA::RelativeTranslationRefinerOptions opt_reltrans;
  MultiPositionRefinerOptions opt_mpr;
  MultiBundleAdjusterOptions opt_ba;
  TriangulatorOptions opt_triangulator;

  // Inlier thresholds for each component
  InlierThresholdOptions inlier_thresholds;

  // Control the number of iterations for each component
  int num_iteration_bundle_adjustment = 3;
  int num_iteration_retriangulation = 1;

  // Control the flow of the global sfm
  bool skip_preprocessing = false;
  bool skip_view_graph_calibration = false;
  bool skip_relative_pose_estimation = false;
  bool skip_rotation_averaging = false;
  bool skip_track_establishment = false;
  bool skip_translation_refinement = false;
  bool skip_position_averaging = false;
  bool skip_finding_new_pairs = false;
  bool skip_bundle_adjustment = false;
  bool skip_retriangulation = true;
  bool skip_pruning = true;

  bool hybrid_translation_averaging = true;
};

class MultiGlobalMapper {
 public:
  MultiGlobalMapper(const MultiGlobalMapperOptions& options)
      : options_(options) {}

  bool Solve(const colmap::Database& database,
             ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks);

 private:
  MultiGlobalMapperOptions options_;
};
}  // namespace MGSfM
