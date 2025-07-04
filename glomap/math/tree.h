
#include "glomap/estimators/multi_global_rotation_averaging.h"
#include "glomap/scene/types_sfm.h"

#include <unordered_set>

namespace glomap {
enum WeightType { INLIER_NUM, INLIER_RATIO };

// Return the number of nodes in the tree
int BFS(const std::vector<std::vector<int>>& graph,
        int root,
        std::vector<int>& parents,
        std::vector<std::pair<int, int>> banned_edges = {});

image_t MaximumSpanningTree(const ViewGraph& view_graph,
                            const std::unordered_map<image_t, Image>& images,
                            std::unordered_map<image_t, image_t>& parents,
                            WeightType type);

image_t RigMaximumSpanningTree(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<image_t, image_t>& parents,
    const std::unordered_set<image_t>& ref_image_ids,
    const std::vector<MGSfM::RefImagePairInfo>& median_rig_pairs);
}  // namespace glomap