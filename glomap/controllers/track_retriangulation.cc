#include "glomap/controllers/track_retriangulation.h"

#include "glomap/io/colmap_converter.h"

#include <colmap/controllers/incremental_pipeline.h>
#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/estimators/cost_functions.h>
#include <colmap/scene/database_cache.h>

#include <set>

namespace glomap {

namespace {
void ParallelSolve(
    const std::shared_ptr<class colmap::Reconstruction>& reconstruction_,
    const ceres::Solver::Options& options) {
  const std::unordered_set<colmap::point3D_t>& point3D_ids =
      reconstruction_->Point3DIds();
  std::vector<struct colmap::Point3D*> points_vec;
  points_vec.reserve(point3D_ids.size());
  for (auto& point3D_id : point3D_ids) {
    if (!reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    struct colmap::Point3D& point3D = reconstruction_->Point3D(point3D_id);

    if (point3D.track.Length() < 2) {
      continue;
    }
    points_vec.emplace_back(&point3D);
  }

  ceres::Solver::Options parallel_option = options;
  parallel_option.linear_solver_type = ceres::DENSE_QR;
  parallel_option.num_threads = 1;
  parallel_option.minimizer_progress_to_stdout = false;
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
#pragma omp parallel for schedule(dynamic)
  for (const auto point_ptr : points_vec) {
    ceres::Problem local_problem(problem_options);

    for (const auto& track_el : point_ptr->track.Elements()) {
      const colmap::Image& image = reconstruction_->Image(track_el.image_id);
      struct colmap::Camera& camera = reconstruction_->Camera(image.CameraId());
      const colmap::Point2D& point2D = image.Point2D(track_el.point2D_idx);
      Eigen::Vector3d pt_calc = image.CamFromWorld() * point_ptr->xyz;
      if (pt_calc(2) < 1e-3) {
        continue;
      }
      local_problem.AddResidualBlock(
          colmap::CreateCameraCostFunction<
              colmap::ReprojErrorConstantPoseCostFunctor>(
              camera.model_id, point2D.xy, image.CamFromWorld()),
          nullptr,
          point_ptr->xyz.data(),
          camera.params.data());
      local_problem.SetParameterBlockConstant(camera.params.data());
    }

    if (local_problem.NumResidualBlocks() >= 2) {
      ceres::Solver::Summary summary;
      ceres::Solve(parallel_option, &local_problem, &summary);
    }
  }
}
}  // namespace

bool RetriangulateTracks(const TriangulatorOptions& options,
                         const colmap::Database& database,
                         std::unordered_map<camera_t, Camera>& cameras,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks) {
  // Following code adapted from COLMAP
  auto database_cache =
      colmap::DatabaseCache::Create(database,
                                    options.min_num_matches,
                                    false,  // ignore_watermarks
                                    {}      // reconstruct all possible images
      );

  // Check whether the image is in the database cache. If not, set the image
  // as not registered to avoid memory error.
  std::vector<image_t> image_ids_notconnected;
  for (auto& image : images) {
    if (!database_cache->ExistsImage(image.first) &&
        image.second.is_registered) {
      image.second.is_registered = false;
      image_ids_notconnected.push_back(image.first);
    }
  }

  // Convert the glomap data structures to colmap data structures
  std::shared_ptr<colmap::Reconstruction> reconstruction_ptr =
      std::make_shared<colmap::Reconstruction>();
  ConvertGlomapToColmap(cameras,
                        images,
                        std::unordered_map<track_t, Track>(),
                        *reconstruction_ptr);

  colmap::IncrementalPipelineOptions options_colmap;
  options_colmap.triangulation.complete_max_reproj_error =
      options.tri_complete_max_reproj_error;
  options_colmap.triangulation.merge_max_reproj_error =
      options.tri_merge_max_reproj_error;
  options_colmap.triangulation.min_angle = options.tri_min_angle;

  reconstruction_ptr->DeleteAllPoints2DAndPoints3D();
  reconstruction_ptr->TranscribeImageIdsToDatabase(database);

  colmap::IncrementalMapper mapper(database_cache);
  mapper.BeginReconstruction(reconstruction_ptr);

  // Triangulate all images.
  const auto tri_options = options_colmap.Triangulation();
  const auto mapper_options = options_colmap.Mapper();

  const std::set<image_t>& reg_image_ids = reconstruction_ptr->RegImageIds();
  const uint32_t print_id = std::max<uint32_t>(reg_image_ids.size() / 10, 1);
  int i = 0;
  for (const auto& image_id : reg_image_ids) {
    if (i % print_id == 0 || i == reg_image_ids.size() - 1)
      std::cout << "\r Triangulating image " << i + 1 << " / "
                << reg_image_ids.size() << std::flush;

    const auto& image = reconstruction_ptr->Image(image_id);

    int num_tris = mapper.TriangulateImage(tri_options, image_id);
    i++;
  }
  std::cout << std::endl;

  // Merge and complete tracks.
  mapper.CompleteAndMergeTracks(tri_options);

  auto ba_options = options_colmap.GlobalBundleAdjustment();
  ba_options.refine_focal_length = false;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = false;
  ba_options.refine_extrinsics = false;

  // Configure bundle adjustment.
  colmap::BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  colmap::ObservationManager observation_manager(*reconstruction_ptr);

  for (int i = 0; i < options_colmap.ba_global_max_refinements; ++i) {
    std::cout << "\r Global bundle adjustment iteration " << i + 1 << " / "
              << options_colmap.ba_global_max_refinements << std::flush;
    // Avoid degeneracies in bundle adjustment.
    observation_manager.FilterObservationsWithNegativeDepth();

    const size_t num_observations =
        reconstruction_ptr->ComputeNumObservations();

    if (true)
      ParallelSolve(reconstruction_ptr,
                    options_colmap.GlobalBundleAdjustment().solver_options);
    else {
      std::unique_ptr<colmap::BundleAdjuster> bundle_adjuster;
      bundle_adjuster = CreateDefaultBundleAdjuster(
          ba_options, ba_config, *reconstruction_ptr);
      if (bundle_adjuster->Solve().termination_type == ceres::FAILURE) {
        return false;
      }
    }

    size_t num_changed_observations = 0;
    num_changed_observations += mapper.CompleteAndMergeTracks(tri_options);
    num_changed_observations += mapper.FilterPoints(mapper_options);
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    if (changed < options_colmap.ba_global_max_refinement_change) {
      break;
    }
  }
  std::cout << std::endl;

  // Add the removed images to the reconstruction
  for (const auto& image_id : image_ids_notconnected) {
    images[image_id].is_registered = true;
    colmap::Image image_colmap;
    ConvertGlomapToColmapImage(images[image_id], image_colmap, true);
    reconstruction_ptr->AddImage(std::move(image_colmap));
  }

  // Convert the colmap data structures back to glomap data structures
  ConvertColmapToGlomap(*reconstruction_ptr, cameras, images, tracks);

  return true;
}

}  // namespace glomap
