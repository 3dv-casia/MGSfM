#include "glomap/controllers/multi_global_mapper.h"

#include "glomap/controllers/option_manager.h"
#include "glomap/io/colmap_io.h"
#include "glomap/types.h"

#include <colmap/util/file.h>
#include <colmap/util/misc.h>
#include <colmap/util/timer.h>

namespace MGSfM {
using namespace glomap;
int RunMGSfM(int argc, char** argv) {
  std::string database_path;
  std::string output_path;

  std::string image_path = "";
  std::string ra_weight_type = "HALF_NORM";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddDefaultOption(
      "ra_weight_type", &ra_weight_type, "{GEMAN_MCCLURE, HALF_NORM}");
  options.AddMultiGlobalMapperFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsFile(database_path)) {
    LOG(ERROR) << "`database_path` is not a file";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "image path: " << image_path;

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  if (ra_weight_type == "HALF_NORM") {
    options.multi_mapper->opt_mre.weight_type =
        MultiRotationEstimatorOptions::HALF_NORM;
    options.multi_mapper->opt_ra.weight_type =
        RotationEstimatorOptions::HALF_NORM;
  } else if (ra_weight_type == "GEMAN_MCCLURE") {
    options.multi_mapper->opt_mre.weight_type =
        MultiRotationEstimatorOptions::GEMAN_MCCLURE;
    options.multi_mapper->opt_ra.weight_type =
        RotationEstimatorOptions::GEMAN_MCCLURE;
  } else {
    LOG(ERROR) << "Invalid constriant type";
    return EXIT_FAILURE;
  }

  // Load the database
  ViewGraph view_graph;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  const colmap::Database database(database_path);

  ConvertDatabaseToGlomap(database, view_graph, cameras, images);

  MultiGlobalMapper multi_global_mapper(*options.multi_mapper);

  // Main solver
  colmap::Timer run_timer;
  run_timer.Start();

  multi_global_mapper.Solve(database, view_graph, cameras, images, tracks);

  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(
      output_path, cameras, images, tracks, output_format, image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

}  // namespace MGSfM
