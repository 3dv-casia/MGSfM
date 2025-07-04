#!/usr/bin/env bash
set -euo pipefail

# 1. download dataset
mkdir -p data/
wget --no-clobber -P data/ https://www.eth3d.net/data/terrains_rig_undistorted.7z
file-roller --extract-to="data/" "data/terrains_rig_undistorted.7z"

# 2. download vocab_tree
mkdir -p data/
wget --no-clobber -P data/ \
     https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin
wget --no-clobber -P data/ \
     https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words256K.bin

# For colmap version >= 3.12, use FAISS-based vocabulary
vocab_tree_path="data/vocab_tree_faiss_flickr100K_words256K.bin"
# else
# vocab_tree_path="vocab_tree_flickr100K_words256K.bin"


# 3. calculating feature correspondences
data_dir="data/terrains/"
image_dir="${data_dir}images/"
database_path="${data_dir}database.db"

# camera setup
cameras_num=4
cameras_folder=( \
  "images_rig_cam4_undistorted" \
  "images_rig_cam5_undistorted" \
  "images_rig_cam6_undistorted" \
  "images_rig_cam7_undistorted" \
)
cameras_models=(PINHOLE PINHOLE PINHOLE PINHOLE)
cameras_params=( \
  "530.632, 528.715, 493.189, 280.36" \
  "527.696, 525.705, 480.633, 304.919" \
  "694.274, 692.970, 391.959, 242.391" \
  "696.477, 695.227, 373.646, 231.643" \
)

# prepare temp directory for per-camera extraction
mkdir -p "${image_dir}tmp/"

for (( i=0; i<cameras_num; i++ )); do
    src="${image_dir}${cameras_folder[i]}"
    dst="${image_dir}tmp/${cameras_folder[i]}"

    mv "$src" "$dst"

    colmap feature_extractor \
        --database_path "$database_path" \
        --image_path "${image_dir}tmp/" \
        --ImageReader.single_camera_per_folder 1 \
        --ImageReader.camera_model "${cameras_models[i]}" \
        --ImageReader.camera_params "${cameras_params[i]}"

    mv "$dst" "$src"
done

rm -rf "${image_dir}tmp/"

# match features with sequential + vocabulary-tree
colmap sequential_matcher \
    --database_path "$database_path" \
    --SequentialMatching.overlap 30 \
    --SequentialMatching.quadratic_overlap 0 \
    --SequentialMatching.loop_detection 1 \
    --SequentialMatching.vocab_tree_path "$vocab_tree_path"

# 4. multi-camera mapping
build/glomap/mgsfm multi_mapper \
    --database_path "$database_path" \
    --output_path "${data_dir}result/" \
    --image_path "$image_dir" \
    --skip_retriangulation 1

# 5. visualize the results in COLMAP GUI:
colmap gui --import_path "${data_dir}result/0/" --database_path "$database_path" --image_path "$image_dir"
