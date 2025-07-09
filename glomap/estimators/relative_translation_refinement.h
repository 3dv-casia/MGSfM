#pragma once

#include "glomap/math/rigid3d.h"
#include "glomap/scene/view_graph.h"

#include <Eigen/Core>

namespace HETA {
using namespace glomap;
struct RelativeTranslationRefinerOptions {
  uint32_t min_feature_correspondence_num = 20;

  double lower_sin_angle_threshold = std::sin(DegToRad(0.3));

  double upper_sin_angle_threshold = std::sin(DegToRad(60.0));

  double min_valid_parallax_ratio = 0.25;
  double min_IRLS_inlier_ratio = 0.5;

  // sin(tri_angle) * sin(residual)
  double residual_threshold = std::sin(DegToRad(5.0)) * std::sin(DegToRad(1.0));

  bool use_prior_relative_translation = false;
};

class RelativeTranslationRefiner {
 public:
  explicit RelativeTranslationRefiner(
      glomap::ViewGraph& view_graph,
      std::unordered_map<colmap::image_t, glomap::Image>& images,
      const RelativeTranslationRefinerOptions& options)
      : options_(options), view_graph_(view_graph), images_(images) {}

  void SelectFeatureMatches(void);

 protected:
  bool SolveMatrix(Eigen::MatrixXd& constraint_matrix,
                   Eigen::Vector3d& relative_translation,
                   Eigen::Array<bool, Eigen::Dynamic, 1>& masks);

  const RelativeTranslationRefinerOptions& options_;

  glomap::ViewGraph& view_graph_;
  std::unordered_map<colmap::image_t, glomap::Image>& images_;
};
}  // namespace HETA
