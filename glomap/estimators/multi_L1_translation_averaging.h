#pragma once
#include "glomap/math/constrained_l1_solver.h"
#include "glomap/scene/types_sfm.h"

#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace MGSfM {
using namespace glomap;

class MultiL1TranslationAveraging {
 public:
  struct RefImagePairInfo {
    RefImagePairInfo()
        : pair_id(-1),
          ref_image_id1(-1),
          ref_image_id2(-1),
          rel_translation(Eigen::Vector3d::Zero()),
          ref_rel_b(Eigen::Vector3d::Zero()),
          scale(1) {}

    RefImagePairInfo(image_pair_t pair_id,
                     image_t image_id1,
                     image_t image_id2,
                     Eigen::Vector3d rel_translation,
                     Eigen::Vector3d ref_rel_b,
                     double scale)
        : pair_id(pair_id),
          ref_image_id1(image_id1),
          ref_image_id2(image_id2),
          rel_translation(rel_translation),
          ref_rel_b(ref_rel_b),
          scale(scale) {}
    image_pair_t pair_id;  // pair id from view graph

    image_t ref_image_id1, ref_image_id2;

    Eigen::Vector3d rel_translation, ref_rel_b;

    double scale;
  };
  MultiL1TranslationAveraging(
      ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<camera_t, Rigid3d>& rel_from_refs,
      const std::unordered_map<image_t, image_t>& image_ref_id)
      : view_graph_(view_graph),
        images_(images),
        rel_from_refs_(rel_from_refs),
        image_ref_id_(image_ref_id) {
    std::unordered_map<image_t, uint32_t> images_edges_num;
    for (const auto& [image_id, rig_id] : image_ref_id) {
      const auto image = images.at(image_id);
      if (image.is_registered && image.camera_id == 1) {
        ref_image_ids_.emplace(image_id);
        images_edges_num[image_id] = 0;
      }
    }

    for (auto& [pair_id, image_pair] : view_graph_.image_pairs) {
      if (!image_pair.is_valid) continue;

      image_t ref_image_id1 = image_ref_id.at(image_pair.image_id1);
      image_t ref_image_id2 = image_ref_id.at(image_pair.image_id2);

      images_edges_num.at(ref_image_id1)++;
      images_edges_num.at(ref_image_id2)++;

      const Eigen::Vector3d rel_translation =
          (images.at(image_pair.image_id2).cam_from_world.rotation.inverse() *
           image_pair.cam2_from_cam1.translation)
              .normalized();
      camera_t camera_id1 = images.at(image_pair.image_id1).camera_id;
      camera_t camera_id2 = images.at(image_pair.image_id2).camera_id;
      const Eigen::Vector3d ref_rel_b =
          images.at(image_pair.image_id2).cam_from_world.rotation.inverse() *
              -rel_from_refs.at(camera_id2).translation -
          images.at(image_pair.image_id1).cam_from_world.rotation.inverse() *
              -rel_from_refs.at(camera_id1).translation;
      double scale = rel_translation.dot(images.at(ref_image_id1).Center() -
                                         images.at(ref_image_id2).Center());

      ref_image_pairs_.emplace(pair_id,
                               RefImagePairInfo(pair_id,
                                                ref_image_id1,
                                                ref_image_id2,
                                                rel_translation,
                                                ref_rel_b,
                                                scale));
      pairs_indices_.emplace_back(pair_id);
    }

    fix_id_ = -1;
    uint32_t max_num = 0;
    for (auto it = images_edges_num.begin(); it != images_edges_num.end();
         ++it) {
      if (it->second > max_num) {
        fix_id_ = it->first;
        max_num = it->second;
      }
    }
    VLOG(2) << "Reference images num: " << ref_image_ids_.size()
            << " Fix image id: " << fix_id_;
  }

  bool MultiEstimataTranslation(void);

 private:
  int MultiSetupL1Solver(Eigen::VectorXd& b);

  ViewGraph& view_graph_;
  std::unordered_map<image_t, Image>& images_;
  std::unordered_map<camera_t, Rigid3d>& rel_from_refs_;
  const std::unordered_map<image_t, image_t>& image_ref_id_;

  std::unordered_set<image_t> ref_image_ids_;
  std::unordered_map<image_t, uint32_t> image_id_to_index_;
  image_t fix_id_;

  std::unordered_map<image_pair_t, RefImagePairInfo> ref_image_pairs_;
  std::vector<image_pair_t> pairs_indices_;

  Eigen::SparseMatrix<double> constraint_matrix_;
  Eigen::VectorXd solution_;
};

}  // namespace MGSfM
