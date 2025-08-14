#include "glomap/controllers/multi_global_mapper.h"

#include "glomap/estimators/bundle_adjustment.h"
#include "glomap/estimators/multi_L1_translation_averaging.h"
#include "glomap/estimators/multi_global_positioning.h"
#include "glomap/estimators/triangulation.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/processors/image_undistorter.h"
#include "glomap/processors/reconstruction_normalizer.h"
#include "glomap/processors/reconstruction_pruning.h"
#include "glomap/processors/relpose_filter.h"
#include "glomap/processors/track_filter.h"
#include "glomap/processors/view_graph_manipulation.h"

#include <colmap/util/file.h>
#include <colmap/util/string.h>
#include <colmap/util/timer.h>

#include <filesystem>

namespace MGSfM {
using namespace glomap;
namespace {
void getResidentMemoryUsage() {
#ifdef __linux__
  std::ifstream statusFile("/proc/self/status");
  std::string line;
  size_t vmRSS = 0;
  size_t vmHWM = 0;

  while (std::getline(statusFile, line)) {
    if (line.find("VmRSS:") == 0) {
      sscanf(line.c_str(), "VmRSS: %zu kB", &vmRSS);
    }
    if (line.find("VmHWM:") == 0) {
      sscanf(line.c_str(), "VmHWM: %zu kB", &vmHWM);
    }
  }

  // Output current resident memory and peak resident memory
  LOG(INFO) << "Current resident memory: "
            << static_cast<double>(vmRSS) / (1024 * 1024) << " GB"
            << " Peak resident memory: "
            << static_cast<double>(vmHWM) / (1024 * 1024) << " GB";
#else
  LOG(INFO) << "Memory usage tracking is only supported on Linux.";
#endif
}

void MultiNormalizeReconstruction(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    std::unordered_map<camera_t, Rigid3d>& rel_from_refs) {
  const auto tform = NormalizeReconstruction(images, tracks);
  for (auto& [camera_id, rel_from_ref] : rel_from_refs) {
    rel_from_ref.translation *= tform.scale;
  }
}
int MultiKeepLargestConnectedComponents(
    ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_map<image_t, image_t>& image_ref_id) {
  ViewGraph ref_view_graph;
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    image_t ref_image_id1 = image_ref_id.at(image_pair.image_id1);
    image_t ref_image_id2 = image_ref_id.at(image_pair.image_id2);
    if (ref_image_id1 == ref_image_id2) continue;
    if (ref_image_id1 > ref_image_id2) std::swap(ref_image_id1, ref_image_id2);
    ref_view_graph.image_pairs.emplace(
        ImagePair::ImagePairToPairId(ref_image_id1, ref_image_id2),
        ImagePair(ref_image_id1, ref_image_id2));
  }
  // Only filter disconnected reference images
  ref_view_graph.KeepLargestConnectedComponents(images);
  // validity of other images depends on their corresponding reference images
  view_graph.num_images = 0;
  for (auto& [image_id, image] : images) {
    image.is_registered = images.at(image_ref_id.at(image_id)).is_registered;
    if (image.is_registered) view_graph.num_images++;
  }

  view_graph.num_pairs = 0;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!images[image_pair.image_id1].is_registered ||
        !images[image_pair.image_id2].is_registered) {
      image_pair.is_valid = false;
    }
    if (image_pair.is_valid) view_graph.num_pairs++;
  }
  return view_graph.num_images;
}
}  // namespace

bool MultiGlobalMapper::Solve(const colmap::Database& database,
                              ViewGraph& view_graph,
                              std::unordered_map<camera_t, Camera>& cameras,
                              std::unordered_map<image_t, Image>& images,
                              std::unordered_map<track_t, Track>& tracks) {
  std::unordered_map<track_t, Track> tracks_full;

  // poses from reference camera to local camera
  std::unordered_map<camera_t, Rigid3d> rel_from_refs;
  // Set camera 1 as reference camera
  for (const auto& [camera_id, camera] : cameras) {
    rel_from_refs[camera_id] = Rigid3d();
  }

  // map image_id to reference image id
  std::unordered_map<image_t, image_t> image_ref_id;
  {
    std::unordered_map<std::string, std::vector<image_t>> image_rigs;
    // Assume that images with identical names belong to the same rig, and their
    // temporal order corresponds to the string order.
    for (const auto& [image_id, image] : images) {
      const std::string rig_name =
          std::filesystem::path(image.file_name).filename().string();
      image_rigs[rig_name].emplace_back(image_id);
    }

    for (const auto& [image_name, image_ids] : image_rigs) {
      image_t ref_id = 0;
      for (const auto& image_id : image_ids) {
        if (images.at(image_id).camera_id == 1) {
          ref_id = image_id;
          break;
        }
      }
      if (ref_id == 0) {
        LOG(WARNING) << "Not all images are available for frame" << image_name;
        continue;
      }
      for (const auto& image_id : image_ids) {
        image_ref_id[image_id] = ref_id;
      }
    }

    LOG(INFO) << "rig numbers: " << image_rigs.size()
              << " camera numbers: " << cameras.size();
  }

  // 0. Preprocessing
  if (!options_.skip_preprocessing) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running preprocessing ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // If camera intrinsics seem to be good, force the pair to use essential
    // matrix
    ViewGraphManipulater::UpdateImagePairsConfig(view_graph, cameras, images);
    ViewGraphManipulater::DecomposeRelPose(view_graph, cameras, images);
    run_timer.PrintSeconds();
  }

  // 1. Run view graph calibration
  if (!options_.skip_view_graph_calibration) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running view graph calibration ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    ViewGraphCalibrator vgcalib_engine(options_.opt_vgcalib);
    if (!vgcalib_engine.Solve(view_graph, cameras, images)) {
      return false;
    }
  }

  // 2. Run relative pose estimation
  if (!options_.skip_relative_pose_estimation) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running relative pose estimation ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // Relative pose relies on the undistorted images
    UndistortImages(cameras, images, true);

    EstimateRelativePosesV2(view_graph, cameras, images, options_.opt_relpose);

    InlierThresholdOptions inlier_thresholds = options_.inlier_thresholds;
    // Undistort the images and filter edges by inlier number
    ImagePairsInlierCount(view_graph, cameras, images, inlier_thresholds, true);

    RelPoseFilter::FilterInlierNum(view_graph,
                                   options_.inlier_thresholds.min_inlier_num);
    RelPoseFilter::FilterInlierRatio(
        view_graph, options_.inlier_thresholds.min_inlier_ratio);

    MultiKeepLargestConnectedComponents(view_graph, images, image_ref_id);

    run_timer.PrintSeconds();
  }

  // 3. Run rotation averaging for three times
  if (!options_.skip_rotation_averaging) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running decoupled rotation averaging ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    colmap::Timer run_timer;
    run_timer.Start();
    if (cameras.size() == 1) {
      RotationEstimator ra_engine(options_.opt_ra);
      // The first run is for filtering
      ra_engine.EstimateRotations(view_graph, images);

      RelPoseFilter::FilterRotations(
          view_graph, images, options_.inlier_thresholds.max_rotation_error);
      if (view_graph.KeepLargestConnectedComponents(images) == 0) {
        LOG(ERROR) << "no connected components are found";
        return false;
      }

      // The second run is for final estimation
      if (!ra_engine.EstimateRotations(view_graph, images)) {
        return false;
      }
      RelPoseFilter::FilterRotations(
          view_graph, images, options_.inlier_thresholds.max_rotation_error);
      image_t num_img = view_graph.KeepLargestConnectedComponents(images);
      if (num_img == 0) {
        LOG(ERROR) << "no connected components are found";
        return false;
      }
      LOG(INFO) << num_img << " / " << images.size()
                << " images are within the connected component." << std::endl;

      run_timer.PrintSeconds();
    } else {
      LOG(INFO) << "Running common rotation averaging ...";

      auto single_view_graph = view_graph;
      single_view_graph.KeepLargestConnectedComponents(images);
      RotationEstimator ra_engine(options_.opt_ra);
      ra_engine.EstimateRotations(single_view_graph, images);
      run_timer.PrintSeconds();

      LOG(INFO) << "Calculating internal rotations ...";

      const int camera_size = rel_from_refs.size();
      std::unordered_map<image_t,
                         std::unordered_map<camera_t, Eigen::Quaterniond>>
          rig_rotations;
      for (const auto& [image_id, rig_id] : image_ref_id) {
        const auto& image = images.at(image_id);
        if (!image.is_registered) continue;
        rig_rotations[rig_id][image.camera_id] = image.cam_from_world.rotation;
      }
      std::unordered_map<camera_t, std::vector<Eigen::Quaterniond>>
          relative_rotations;
      for (const auto& [rig_id, rotations] : rig_rotations) {
        const auto it = rotations.find(1);
        if (it == rotations.end()) continue;
        const auto& base_rotation = it->second;
        for (const auto& [camera_id, rotation] : rotations) {
          if (camera_id == 1) continue;
          relative_rotations[camera_id].emplace_back(rotation *
                                                     base_rotation.inverse());
        }
      }
      std::vector<double> angle_errors;
      for (const auto& [camera_id, rotations] : relative_rotations) {
        if (camera_id == 1) continue;
        rel_from_refs.at(camera_id).rotation =
            MultiRotationEstimator::MedianRotations(rotations, angle_errors);
        CHECK_NE(angle_errors.size(), 0)
            << "Median internal camera rotation estimation failed !";
      }
      // Update the is_registered flag of all images
      MultiKeepLargestConnectedComponents(view_graph, images, image_ref_id);
      run_timer.PrintSeconds();

      LOG(INFO) << "Running multiple rotation averaging ...";
      MultiRotationEstimator mra_engine(options_.opt_mre);
      // The first run is for filtering
      mra_engine.EstimateRotations(
          view_graph, images, rel_from_refs, image_ref_id);

      RelPoseFilter::FilterRotations(
          view_graph, images, options_.inlier_thresholds.max_rotation_error);

      image_t num_img =
          MultiKeepLargestConnectedComponents(view_graph, images, image_ref_id);

      LOG(INFO) << num_img << " / " << images.size()
                << " images are within the connected component." << std::endl;
    }
    run_timer.PrintSeconds();
  }

  // 4. Track establishment and selection
  if (!options_.skip_track_establishment) {
    colmap::Timer run_timer;
    run_timer.Start();

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running track establishment ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    TrackEngine track_engine(view_graph, images, options_.opt_track);
    track_engine.EstablishFullTracksV2(tracks_full);

    // Filter the tracks
    track_engine.FindTracksForProblem(tracks_full, tracks);
    LOG(INFO) << "Filtered tracks num: " << tracks.size() << std::endl;

    std::unordered_set<image_t> track_cover_images;
    for (auto& [track_id, track] : tracks) {
      for (auto& [image_id, feature_id] : track.observations) {
        track_cover_images.emplace(image_id);
      }
    }

    for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
      if (track_cover_images.count(image_pair.image_id1) == 0 ||
          track_cover_images.count(image_pair.image_id2) == 0) {
        image_pair.is_valid = false;
      }
    }
    image_t num_img =
        MultiKeepLargestConnectedComponents(view_graph, images, image_ref_id);
    LOG(INFO) << num_img << " / " << images.size()
              << " images are within the connected component." << std::endl;

    run_timer.PrintSeconds();
  }

  getResidentMemoryUsage();
  // 5. Hybrid position averaging
  if (!options_.skip_position_averaging &&
      options_.hybrid_translation_averaging) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running hybrid translation averaging ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // Undistort images in case all previous steps are skipped
    // Skip images where an undistortion already been done
    UndistortImages(cameras, images, false);
    if (!options_.skip_finding_new_pairs) {
      view_graph.FindPairsFromTracks(tracks);
    }
    if (!options_.skip_translation_refinement) {
      HETA::RelativeTranslationRefiner relative_translation_refinement(
          view_graph, images, options_.opt_reltrans);
      relative_translation_refinement.SelectFeatureMatches();
      image_t num_img =
          MultiKeepLargestConnectedComponents(view_graph, images, image_ref_id);
      LOG(INFO) << num_img << " / " << images.size()
                << " images are within the connected component.";
    }
    run_timer.PrintSeconds();

    MultiL1TranslationAveraging multi_l1_ta(
        view_graph, images, rel_from_refs, image_ref_id);
    multi_l1_ta.MultiEstimataTranslation();

    MultiPositionRefiner mpar_engine(options_.opt_mpr);
    auto& opt_mpr = mpar_engine.GetOptions();
    opt_mpr.constraint_type = MultiPositionRefinerOptions::ONLY_CAMERAS;
    opt_mpr.solver_options.function_tolerance = 1e-5;
    if (!mpar_engine.Solve(
            view_graph, cameras, images, tracks, rel_from_refs, image_ref_id)) {
      return false;
    }
    L1Triangulation(images, tracks);
    MultiNormalizeReconstruction(images, tracks, rel_from_refs);
    run_timer.PrintSeconds();
    opt_mpr.constraint_type = MultiPositionRefinerOptions::ONLY_POINTS;
    opt_mpr.solver_options.function_tolerance =
        options_.opt_mpr.solver_options.function_tolerance;
    if (!mpar_engine.Solve(
            view_graph, cameras, images, tracks, rel_from_refs, image_ref_id)) {
      return false;
    }
    MultiNormalizeReconstruction(images, tracks, rel_from_refs);
    AngleRansacTriangulation(images, cameras, tracks, 1.0, true);
    run_timer.PrintSeconds();
  } else if (!options_.skip_position_averaging &&
             !options_.hybrid_translation_averaging) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running multiple global positioning ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();
    // Undistort images in case all previous steps are skipped
    // Skip images where an undistortion already been done
    UndistortImages(cameras, images, false);
    MultiGlobalPositioner mgp_engine(MultiGlobalPositionerOptions{});
    auto& opt_mgp = mgp_engine.GetOptions();
    if (!mgp_engine.Solve(
            view_graph, cameras, images, tracks, rel_from_refs, image_ref_id)) {
      return false;
    }
    TrackFilter::FilterTracksByAngle(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_angle_error);
    run_timer.PrintSeconds();
  }

  getResidentMemoryUsage();

  if (!options_.skip_bundle_adjustment) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running multiple bundle adjustment ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    LOG(INFO) << "Bundle adjustment start" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();

    for (int ite = 0; ite < options_.num_iteration_bundle_adjustment; ite++) {
      MultiBundleAdjuster ba_engine(options_.opt_ba);

      auto& multi_ba_opt = ba_engine.GetOptions();

      // Staged bundle adjustment
      // 6.1. First stage: optimize positions only
      multi_ba_opt.optimize_rotations = false;
      if (!ba_engine.Solve(view_graph,
                           cameras,
                           images,
                           tracks,
                           rel_from_refs,
                           image_ref_id)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 1 finished (position only)";
      run_timer.PrintSeconds();

      // 6.2. Second stage: optimize rotations if desired
      multi_ba_opt.optimize_rotations = options_.opt_ba.optimize_rotations;
      if (multi_ba_opt.optimize_rotations && !ba_engine.Solve(view_graph,
                                                              cameras,
                                                              images,
                                                              tracks,
                                                              rel_from_refs,
                                                              image_ref_id)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 2 finished";
      if (ite != options_.num_iteration_bundle_adjustment - 1)
        run_timer.PrintSeconds();

      MultiNormalizeReconstruction(images, tracks, rel_from_refs);

      // 6.3. Filter tracks based on the estimation
      // For the filtering, in each round, the criteria for outlier is
      // tightened. If only few tracks are changed, no need to start bundle
      // adjustment right away. Instead, use a more strict criteria to
      // filter
      UndistortImages(cameras, images, true);
      LOG(INFO) << "Filtering tracks by reprojection ...";

      bool status = true;
      size_t filtered_num = 0;
      while (status && ite < options_.num_iteration_bundle_adjustment) {
        double scaling =
            std::max(options_.num_iteration_bundle_adjustment - ite, 1);
        filtered_num += TrackFilter::FilterTracksByReprojection(
            view_graph,
            cameras,
            images,
            tracks,
            scaling * options_.inlier_thresholds.max_reprojection_error);

        if (filtered_num > 1e-3 * tracks.size()) {
          status = false;
        } else
          ite++;
      }
      if (status) {
        LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
        break;
      }
    }
    // Filter tracks based on the estimation
    UndistortImages(cameras, images, true);
    LOG(INFO) << "Filtering tracks by reprojection ...";
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_reprojection_error);
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);

    run_timer.PrintSeconds();
  }

  // 7. Retriangulation
  if (!options_.skip_retriangulation) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running retriangulation ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    for (int ite = 0; ite < options_.num_iteration_retriangulation; ite++) {
      colmap::Timer run_timer;
      run_timer.Start();
      RetriangulateTracks(
          options_.opt_triangulator, database, cameras, images, tracks);

      run_timer.PrintSeconds();

      std::cout << "-------------------------------------" << std::endl;
      std::cout << "Running multiple bundle adjustment ..." << std::endl;
      std::cout << "-------------------------------------" << std::endl;
      LOG(INFO) << "Bundle adjustment start" << std::endl;
      MultiBundleAdjuster mba_engine(MultiBundleAdjusterOptions{});
      auto& mba_option = mba_engine.GetOptions();
      if (!mba_engine.Solve(view_graph,
                            cameras,
                            images,
                            tracks,
                            rel_from_refs,
                            image_ref_id)) {
        return false;
      }
      // Filter tracks based on the estimation
      UndistortImages(cameras, images, true);
      LOG(INFO) << "Filtering tracks by reprojection ...";
      TrackFilter::FilterTracksByReprojection(
          view_graph,
          cameras,
          images,
          tracks,
          options_.inlier_thresholds.max_reprojection_error);
      if (!mba_engine.Solve(view_graph,
                            cameras,
                            images,
                            tracks,
                            rel_from_refs,
                            image_ref_id)) {
        return false;
      }
      run_timer.PrintSeconds();
    }

    MultiNormalizeReconstruction(images, tracks, rel_from_refs);

    // Filter tracks based on the estimation
    UndistortImages(cameras, images, true);

    LOG(INFO) << "Filtering tracks by reprojection ...";
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_reprojection_error);
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);
  }
  getResidentMemoryUsage();
  // 8. Reconstruction pruning
  if (!options_.skip_pruning) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Running postprocessing ..." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    colmap::Timer run_timer;
    run_timer.Start();

    // Prune weakly connected images
    PruneWeaklyConnectedImages(images, tracks);

    run_timer.PrintSeconds();
  }

  return true;
}
}  // namespace MGSfM