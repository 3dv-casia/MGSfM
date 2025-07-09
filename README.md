[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2507.03306-B31B1B.svg)](https://arxiv.org/pdf/2507.03306)

# MGSfM: Multi-Camera Driven Global Structure-from-Motion
A fast, robust, and scalable global Structure-from-Motion (SfM) pipeline for sequential image collections captured by single- or multi-camera rigs.

## **[Paper](https://arxiv.org/abs/2507.03306)** 


## 📖 Overview

MGSfM builds on [GLOMAP](https://github.com/colmap/glomap) and takes a COLMAP database as input to produce a COLMAP‐compatible sparse reconstruction. Compared to GLOMAP, MGSfM offers:

* **Hybrid translation averaging:** Hybrid algorithm combining efficiency and robustness for rapid, accurate global translation estimation, while using less resident memory.
* **Supports flexible rigs:** Works with single-camera or multi-camera setups, overlapping or non-overlapping fields of view, panoramic or standard systems.
* **Automates rig calibration:** Requires no prior internal pose information—handles both the rig unit and individual camera poses seamlessly.

If you use MGSfM in your work, please cite:

```bibtex
@inproceedings{tao2025mgsfm,
  author    = {Tao, Peilin and Cui, Hainan and Tu, Diantao and Shen, Shuhan},
  title     = {{MGSfM: Multi-Camera Driven Global Structure-from-Motion}},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2025},
}

@inproceedings{tao2024revisiting,
  author    = {Tao, Peilin and Cui, Hainan and Rong, Mengqi and Shen, Shuhan},
  title     = {Revisiting Global Translation Estimation with Feature Tracks},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024},
}
```


## 🚀 Quick Start

### 1. Installation

1. **Install COLMAP dependencies** using the [COLMAP build guide](https://colmap.github.io/install.html#build-from-source).
2. **Clone and build MGSfM:**
```bash
git clone https://github.com/3dv-casia/MGSfM.git
cd MGSfM
mkdir build && cd build
cmake .. -GNinja
ninja && sudo ninja install
```
**CMake requirement:** MGSfM uses CMake’s `FetchContent` to automatically download COLMAP & PoseLib, which requires CMake ≥ 3.28 (as in [GLOMAP](https://github.com/colmap/glomap)).

* **Upgrade CMake:**

  ```bash
  wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1.tar.gz
  tar xzf cmake-3.30.1.tar.gz && cd cmake-3.30.1
  ./bootstrap && make -j$(nproc) && sudo make install
  ```
* **Or disable automatic fetching and use a self-installed version of COLMAP & PoseLib; this requires CMake ≥ 3.10:**

  ```bash
  cd build
  cmake .. -DFETCH_COLMAP=OFF -DFETCH_POSELIB=OFF
  ```

### 2. Prepare Your Data

Organize images in subfolders per camera, with matching filenames for each frame:

```
images/
  cam1/img0001.jpg
       /img0002.jpg
  cam2/img0001.jpg
       /img0002.jpg
  ...
```

> **Note**: Only a single rig is supported currently; multi-rig support is planned for a future release.

### 3. Build the COLMAP Database

```bash
colmap feature_extractor \
  --image_path images/ \
  --database_path database.db \
  --ImageReader.single_camera_per_folder 1

colmap sequential_matcher \
  --database_path database.db \
  --SequentialMatching.loop_detection 1 \
  --SequentialMatching.vocab_tree_path vocab_tree.bin
```

### 4. Run MGSfM

```bash
mgsfm multi_mapper \
  --database_path database.db \
  --image_path images/ \
  --output_path output/
```

Use `mgsfm multi_mapper -h` for full CLI details.

### 5. Demo
Run the provided demo on an ETH3D rig dataset:
```bash
cd MGSfM
mkdir -p data
wget -P data/ https://www.eth3d.net/data/terrains_rig_undistorted.7z
7z x data/terrains_rig_undistorted.7z -odata/
bash demo.sh
```

## 🤝 Contributing

Bug reports, feature requests, and pull requests are welcome! Please open an issue or submit a PR on GitHub.


## 🎗️ Acknowledgements

This work leverages ideas and code from GLOMAP, HETA, COLMAP, PoseLib, and Theia. Please cite these projects if they contribute to your results.


## 📜 License

```
Copyright (c) 2024, ETH Zurich.
Copyright (c) 2025, Peilin Tao.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of ETH Zurich nor the names of its contributors may
      be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
