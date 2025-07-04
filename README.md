# MGSfM: Multi-Camera Driven Global Structure-from-Motion

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2507.20219-B31B1B.svg)](https://arxiv.org/pdf/2507.20219)

---

## Table of Contents
- [MGSfM: Multi-Camera Driven Global Structure-from-Motion](#mgsfm-multi-camera-driven-global-structure-from-motion)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
    - [Directory Layout](#directory-layout)
    - [Build COLMAP Database](#build-colmap-database)
    - [Run Multi-Camera SfM](#run-multi-camera-sfm)
    - [Simple Example](#simple-example)
  - [Contributing](#contributing)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)

---

## About

**MGSfM** is an efficient, robust, and scalable global Structure-from-Motion pipeline designed for sequential images captured by multi-camera or single-camera systems. Building on [GLOMAP](https://github.com/colmap/glomap), it takes a COLMAP database as input and outputs a COLMAP-format sparse reconstruction. Compared to GLOMAP, MGSfM delivers faster and more robust translation averaging, with equal or superior reconstruction quality, while consuming less resident memory. The features of MGSfM are:
* **Hybrid Translation Averaging**: Combines efficiency and robustness in a single algorithm, delivering fast and accurate global translation estimation for sequential image inputs.
* **Automatic Rig Calibration**: Requires no prior knowledge of internal camera poses, seamlessly handling rigid unit poses and internal camera poses.
* **Flexible Rig Configurations**: Supports any setup—single-camera or multi-camera rigs, overlapping or non-overlapping fields of view, and panoramic or non-panoramic systems.


If you use MGSfM in your research, please cite:

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

---
## Installation

1. **Install COLMAP dependencies** per the [COLMAP build guide](https://colmap.github.io/install.html#build-from-source).
2. **Clone and build MGSfM**:

   ```bash
   git clone https://github.com/3dv-casia/MGSfM.git
   cd MGSfM
   mkdir build && cd build
   cmake .. -GNinja
   ninja && ninja install
   ```
3. **(Optional)** Disable automatic fetching of COLMAP or PoseLib:

   ```bash
   cmake .. -DFETCH_COLMAP=OFF -DFETCH_POSELIB=OFF
   ```

> **Note:** CMake ≥ 3.28 is required for `FetchContent`. To build with an older system CMake, install CMake 3.30.1:
>
> ```bash
> wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1.tar.gz
> tar xzf cmake-3.30.1.tar.gz && cd cmake-3.30.1
> ./bootstrap && make -j$(nproc) && sudo make install
> ```
---
## Quick Start

### Directory Layout

Organize images by camera subfolders. Filenames must match across cameras for the same frame:

```
images/
  cam1/
    img0001.jpg
    img0002.jpg
  cam2/
    img0001.jpg
    img0002.jpg
  ...
```
Currently, the project supports only a single rig as input, which covers most multi-camera datasets.
The version for multiple rigs will be released soon.
### Build COLMAP Database

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

### Run Multi-Camera SfM

```bash
mgsfm multi_mapper \
    --database_path database.db \
    --image_path images/ \
    --output_path output/
```

For more efficient:

```bash
mgsfm multi_mapper \
    --database_path database.db \
    --image_path images/ \
    --output_path output/ \
    --MultiplePositionRefinement.function_tolerance 1e-3 \
    --ba_iteration_num 2 \
    --skip_retriangulation 1
```

For more details on the command line interface, use `mgsfm multi_mapper -h` for full CLI options.

### Simple Example

Run the provided demo on an ETH3D rig dataset:

```bash
bash demo.sh
```
---
## Contributing

Contributions (bug reports, bug fixes, improvements, etc.) are very welcome and should be submitted in the form of new issues and/or pull requests on GitHub.

---
## Acknowledgements

This project builds upon the insights and code from GLOMAP, HETA, COLMAP, PoseLib, and Theia. Please cite those if you use MGSfM.

---
## License

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
