#pragma once

#include "glomap/math/l1_solver_vec3.h"
#include "glomap/scene/types.h"

#include <colmap/estimators/cost_functions.h>
#include <colmap/estimators/triangulation.h>
#include <colmap/geometry/pose.h>

namespace MGSfM {
using namespace glomap;
int AngleRansacTriangulation(
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<track_t, Track>& tracks,
    double max_angle_error,
    bool is_filter = true) {
  LOG(INFO) << "Angle RANSAC Triangulation!";
  std::vector<track_t> tracks_vec;

  tracks_vec.reserve(tracks.size());
  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < 3) {
      track.observations.clear();
      continue;
    }
    tracks_vec.emplace_back(track_id);
  }

  colmap::EstimateTriangulationOptions options_;
  options_.residual_type =
      colmap::TriangulationEstimator::ResidualType::ANGULAR_ERROR;
  options_.ransac_options.max_error = DegToRad(max_angle_error);
  options_.ransac_options.confidence = 0.9999;
  options_.ransac_options.min_inlier_ratio = 0.02;
  options_.ransac_options.max_num_trials = 10000;
  size_t counter = 0;
#pragma omp parallel for schedule(dynamic)
  for (const auto& track_id : tracks_vec) {
    auto& track = tracks.at(track_id);
    const uint32_t images_num = track.observations.size();
    std::vector<Eigen::Vector2d> points;
    points.resize(images_num);
    std::vector<Rigid3d const*> cams_from_world;
    cams_from_world.resize(images_num);
    std::vector<colmap::Camera const*> track_cameras;
    track_cameras.resize(images_num);

    for (int i = 0; i < images_num; i++) {
      const auto& observation = track.observations.at(i);
      const auto& image = images.at(observation.first);
      points[i] = image.features[observation.second];
      cams_from_world[i] = &image.cam_from_world;
      track_cameras[i] =
          static_cast<colmap::Camera const*>(&cameras.at(image.camera_id));
    }

    std::vector<char> inlier_mask;
    if (colmap::EstimateTriangulation(options_,
                                      points,
                                      cams_from_world,
                                      track_cameras,
                                      &inlier_mask,
                                      &track.xyz)) {
      track.is_initialized = true;
      std::vector<Observation> observation_new;
      observation_new.reserve(track.observations.size());
      for (int i = 0; i < inlier_mask.size(); i++) {
        if (inlier_mask[i]) {
          observation_new.emplace_back(track.observations[i]);
        }
      }
      if (observation_new.size() != track.observations.size() && is_filter) {
#pragma omp atomic
        counter++;
        track.observations = observation_new;
      }
    } else {
      track.observations.clear();
#pragma omp atomic
      counter++;
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by RANSAC with angle: " << max_angle_error;
  return counter;
}

void L1Triangulation(const std::unordered_map<image_t, Image>& images,
                     std::unordered_map<track_t, Track>& tracks) {
  LOG(INFO) << "L1 Cross Triangulation ...";
  std::vector<track_t> tracks_vec;

  tracks_vec.reserve(tracks.size());
  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < 3) {
      track.observations.clear();
      continue;
    }
    tracks_vec.emplace_back(track_id);
  }

  HETA::L1SolverVec3Options options;
#pragma omp parallel for schedule(dynamic)
  for (const auto& track_id : tracks_vec) {
    auto& track = tracks.at(track_id);
    const uint32_t images_num = track.observations.size();
    Eigen::MatrixXd mat(images_num * 3, 3);
    Eigen::VectorXd rhs(images_num * 3);

    for (int i = 0; i < images_num; i++) {
      const auto& observation = track.observations.at(i);
      const auto& image = images.at(observation.first);
      const Eigen::Vector3d& feature_undist =
          image.features_undist[observation.second];
      if (feature_undist.array().isNaN().any()) {
        continue;
      }
      Eigen::Vector3d point = (image.cam_from_world.rotation.inverse() *
                               image.features_undist[observation.second])
                                  .normalized();
      const auto position = image.Center();
      mat.block<3, 3>(3 * i, 0) = colmap::CrossProductMatrix(point);
      rhs.segment<3>(3 * i) = point.cross(position).transpose();
    }
    HETA::L1SolverVec3 solver(options, mat);
    solver.Solve(rhs, &track.xyz);
    track.is_initialized = true;
  }
}
}  // namespace MGSfM