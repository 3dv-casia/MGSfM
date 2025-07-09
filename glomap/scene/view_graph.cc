#include "glomap/scene/view_graph.h"

#include "glomap/math/union_find.h"

#include <queue>

namespace glomap {

int ViewGraph::KeepLargestConnectedComponents(
    std::unordered_map<image_t, Image>& images) {
  EstablishAdjacencyList();

  int num_comp = FindConnectedComponent();

  int max_idx = -1;
  int max_img = 0;
  for (int comp = 0; comp < num_comp; comp++) {
    if (connected_components[comp].size() > max_img) {
      max_img = connected_components[comp].size();
      max_idx = comp;
    }
  }

  if (max_img == 0) return 0;

  std::unordered_set<image_t> largest_component = connected_components[max_idx];

  // Set all images to not registered
  for (auto& [image_id, image] : images) image.is_registered = false;

  // Set the images in the largest component to registered
  for (auto image_id : largest_component) images[image_id].is_registered = true;

  // set all pairs not in the largest component to invalid
  num_pairs = 0;
  for (auto& [pair_id, image_pair] : image_pairs) {
    if (!images[image_pair.image_id1].is_registered ||
        !images[image_pair.image_id2].is_registered) {
      image_pair.is_valid = false;
    }
    if (image_pair.is_valid) num_pairs++;
  }

  num_images = largest_component.size();
  return max_img;
}

int ViewGraph::FindConnectedComponent() {
  connected_components.clear();
  std::unordered_map<image_t, bool> visited;
  for (auto& [image_id, neighbors] : adjacency_list) {
    visited[image_id] = false;
  }

  for (auto& [image_id, neighbors] : adjacency_list) {
    if (!visited[image_id]) {
      std::unordered_set<image_t> component;
      BFS(image_id, visited, component);
      connected_components.push_back(component);
    }
  }

  return connected_components.size();
}

int ViewGraph::MarkConnectedComponents(
    std::unordered_map<image_t, Image>& images, int min_num_img) {
  EstablishAdjacencyList();

  int num_comp = FindConnectedComponent();

  std::vector<std::pair<int, int>> cluster_num_img(num_comp);
  for (int comp = 0; comp < num_comp; comp++) {
    cluster_num_img[comp] =
        std::make_pair(connected_components[comp].size(), comp);
  }
  std::sort(cluster_num_img.begin(), cluster_num_img.end(), std::greater<>());

  // Set the cluster number of every image to be -1
  for (auto& [image_id, image] : images) image.cluster_id = -1;

  int comp = 0;
  for (; comp < num_comp; comp++) {
    if (cluster_num_img[comp].first < min_num_img) break;
    for (auto image_id : connected_components[cluster_num_img[comp].second])
      images[image_id].cluster_id = comp;
  }

  return comp;
}

void ViewGraph::BFS(image_t root,
                    std::unordered_map<image_t, bool>& visited,
                    std::unordered_set<image_t>& component) {
  std::queue<image_t> q;
  q.push(root);
  visited[root] = true;
  component.insert(root);

  while (!q.empty()) {
    image_t curr = q.front();
    q.pop();

    for (image_t neighbor : adjacency_list[curr]) {
      if (!visited[neighbor]) {
        q.push(neighbor);
        visited[neighbor] = true;
        component.insert(neighbor);
      }
    }
  }
}

void ViewGraph::EstablishAdjacencyList() {
  adjacency_list.clear();
  for (auto& [pair_id, image_pair] : image_pairs) {
    if (image_pair.is_valid) {
      adjacency_list[image_pair.image_id1].insert(image_pair.image_id2);
      adjacency_list[image_pair.image_id2].insert(image_pair.image_id1);
    }
  }
}

void ViewGraph::FindPairsFromTracks(
    const std::unordered_map<track_t, Track>& tracks) {
  // Find new image pairs from feature tracks
  constexpr double min_overlap = 0.1;
  constexpr int min_inlier_num = 30;
  EstablishAdjacencyList();
  std::unordered_map<image_t, std::vector<track_t>> image_tracks;
  std::unordered_map<image_pair_t, std::vector<Eigen::Vector2i>>
      new_image_pairs;

  for (const auto& [track_id, track] : tracks) {
    const auto& observations = track.observations;
    const int track_size = observations.size();
    for (int i = 0; i < track_size; ++i) {
      image_t image_id1 = observations[i].first;
      feature_t feature_id1 = observations[i].second;
      image_tracks[image_id1].emplace_back(track_id);
      const auto& adjacent_images = adjacency_list.at(image_id1);
      for (int k = i + 1; k < track_size; ++k) {
        image_t image_id2 = observations[k].first;
        if (adjacent_images.count(image_id2) || image_id1 == image_id2)
          continue;
        feature_t feature_id2 = observations[k].second;
        image_pair_t pair_id =
            ImagePair::ImagePairToPairId(image_id1, image_id2);
        if (image_id1 < image_id2)
          new_image_pairs[pair_id].emplace_back(feature_id1, feature_id2);
        else
          new_image_pairs[pair_id].emplace_back(feature_id2, feature_id1);
      }
    }
  }

  int new_pairs_num = 0;
  for (const auto& [pair_id, matches] : new_image_pairs) {
    const int matches_num = matches.size();

    if (matches_num < min_inlier_num) continue;
    image_t image_id1, image_id2;
    ImagePair::PairIdToImagePair(pair_id, image_id1, image_id2);
    if (static_cast<int>(std::min(image_tracks.at(image_id1).size(),
                                  image_tracks.at(image_id2).size()) *
                         min_overlap) > matches_num)
      continue;
    new_pairs_num++;

    ImagePair& image_pair =
        image_pairs.emplace(pair_id, ImagePair(image_id1, image_id2))
            .first->second;
    image_pair.matches = Eigen::MatrixXi(matches_num, 2);
    image_pair.inliers.resize(matches_num);
    for (int i = 0; i < matches_num; ++i) {
      image_pair.matches.row(i) = matches.at(i);
      image_pair.inliers[i] = i;
    }
  }
  LOG(INFO) << "Find new image pairs num: " << new_pairs_num;
}
}  // namespace glomap
