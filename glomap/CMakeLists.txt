set(SOURCES
    controllers/global_mapper.cc
    controllers/option_manager.cc
    controllers/rotation_averager.cc
    controllers/track_establishment.cc
    controllers/track_retriangulation.cc
    estimators/bundle_adjustment.cc
    estimators/global_positioning.cc
    estimators/global_rotation_averaging.cc
    estimators/gravity_refinement.cc
    estimators/relpose_estimation.cc
    estimators/view_graph_calibration.cc
    io/colmap_converter.cc
    io/colmap_io.cc
    io/pose_io.cc
    math/gravity.cc
    math/rigid3d.cc
    math/tree.cc
    math/two_view_geometry.cc
    processors/image_pair_inliers.cc
    processors/image_undistorter.cc
    processors/reconstruction_normalizer.cc
    processors/reconstruction_pruning.cc
    processors/relpose_filter.cc
    processors/track_filter.cc
    processors/view_graph_manipulation.cc
    scene/view_graph.cc

    controllers/multi_global_mapper.cc
    estimators/relative_translation_refinement.cc
    estimators/multi_global_rotation_averaging.cc
    estimators/multi_L1_translation_averaging.cc
    estimators/multi_global_position_refinement.cc
    estimators/multi_bundle_adjustment.cc
    estimators/multi_global_positioning.cc
    math/constrained_l1_solver.cc
)

set(HEADERS
    controllers/global_mapper.h
    controllers/option_manager.h
    controllers/rotation_averager.h
    controllers/track_establishment.h
    controllers/track_retriangulation.h
    estimators/bundle_adjustment.h
    estimators/cost_function.h
    estimators/global_positioning.h
    estimators/global_rotation_averaging.h
    estimators/gravity_refinement.h
    estimators/relpose_estimation.h
    estimators/optimization_base.h
    estimators/view_graph_calibration.h
    io/colmap_converter.h
    io/colmap_io.h
    io/pose_io.h
    math/gravity.h
    math/l1_solver.h
    math/rigid3d.h
    math/tree.h
    math/two_view_geometry.h
    math/union_find.h
    processors/image_pair_inliers.h
    processors/image_undistorter.h
    processors/reconstruction_normalizer.h
    processors/reconstruction_pruning.h
    processors/relpose_filter.h
    processors/track_filter.h
    processors/view_graph_manipulation.h
    scene/camera.h
    scene/image_pair.h
    scene/image.h
    scene/track.h
    scene/types_sfm.h
    scene/types.h

    controllers/multi_global_mapper.h
    estimators/relative_translation_refinement.h
    estimators/multi_global_rotation_averaging.h
    estimators/multi_L1_translation_averaging.h
    estimators/multi_global_position_refinement.h
    estimators/multi_bundle_adjustment.h
    estimators/multi_global_positioning.h
    estimators/triangulation.h
    math/constrained_l1_solver.h
    math/l1_solver_vec3.h
)

add_library(glomap ${SOURCES} ${HEADERS})
if(NOT FETCH_COLMAP)
    target_link_libraries(glomap PUBLIC colmap::colmap)
else()
    target_link_libraries(glomap PUBLIC colmap)
endif()

if(NOT FETCH_POSELIB)
    target_link_libraries(glomap PUBLIC PoseLib::PoseLib)
else()
    target_link_libraries(glomap PUBLIC PoseLib)
endif()

target_link_libraries(
    glomap
    PUBLIC
        Eigen3::Eigen
        Ceres::ceres
        SuiteSparse::CHOLMOD
        ${BOOST_LIBRARIES}
)
target_include_directories(glomap PUBLIC ..)

if(OPENMP_FOUND)
    target_link_libraries(glomap PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(glomap PRIVATE /bigobj)
else()
    target_compile_options(glomap PRIVATE
        -Wall
        -Werror
        -Wno-sign-compare
        -Wno-unused-variable
    )
endif()


add_executable(glomap_main
    glomap.cc
    exe/global_mapper.h
    exe/global_mapper.cc
    exe/rotation_averager.h
    exe/rotation_averager.cc
)

target_link_libraries(glomap_main glomap)

set_target_properties(glomap_main PROPERTIES OUTPUT_NAME glomap)
install(TARGETS glomap_main DESTINATION bin)

add_executable(mgsfm
    mgsfm.cc
    exe/multi_global_mapper.h
    exe/multi_global_mapper.cc
)

target_link_libraries(mgsfm glomap)

set_target_properties(mgsfm PROPERTIES OUTPUT_NAME mgsfm)
install(TARGETS mgsfm DESTINATION bin)

if(TESTS_ENABLED)
    add_executable(glomap_test
        controllers/global_mapper_test.cc
        controllers/rotation_averager_test.cc
    )
    target_link_libraries(
        glomap_test
        PRIVATE
            glomap
            GTest::gtest
            GTest::gtest_main)
    add_test(NAME glomap_test COMMAND glomap_test)
endif()
