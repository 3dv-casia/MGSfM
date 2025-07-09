#!/usr/bin/env bash
set -euo pipefail

for data in "KITTI01" "KITTI06"; do
    data_dir="data/"${data}"/"
    database_path="${data_dir}database.db"

    # multi-camera mapping
    # build/glomap/mgsfm multi_mapper \
    #     --database_path "$database_path" \
    #     --output_path "${data_dir}mgsfm_result/" \
    #     --image_path "" \
    #     --ba_iteration_num 2 \
    #     --skip_retriangulation 1

    # visualize the results in COLMAP GUI:
    colmap gui --import_path "${data_dir}mgsfm_result/0/" --database_path "$database_path" --image_path "$data_dir"

    # try multi-camera glomap (DMRA+MGP in paper)
    build/glomap/mgsfm multi_mapper \
        --database_path "$database_path" \
        --output_path "${data_dir}glomap_result/" \
        --image_path "" \
        --ba_iteration_num 2 \
        --skip_retriangulation 1 \
        --hybrid_translation_averaging 0

    # visualize the results in COLMAP GUI:
    colmap gui --import_path "${data_dir}glomap_result/0/" --database_path "$database_path" --image_path "$data_dir"
done
