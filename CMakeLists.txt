cmake_minimum_required(VERSION 2.8.3)
project(extrinsics_calibrator)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  apriltag_tracker
  gtsam_catkin
  pybind11_catkin
)

catkin_package(
  INCLUDE_DIRS
  include 
  CATKIN_DEPENDS
  roscpp
  apriltag_tracker
  gtsam_catkin
  rosbag
  pybind11_catkin
)
find_package(OpenCV REQUIRED)
set(EIGEN3_INCLUDE_DIR "/home/icoderaven/packages/eigen")
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
pkg_check_modules(PYTHON REQUIRED python)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ffast-math -O3 -march=native -Ofast -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -ffast-math -O3 -march=native -Ofast -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -fopenmp -g")

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${YAML_CPP_INCLUDEDIR}
  ${PYBIND11_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${gtsam_catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

message("GTSAM_CATKIN Includes : ${gtsam_catkin_INCLUDE_DIRS}")
message("GTSAM_CATKIN Libraries : ${gtsam_catkin_LIBRARIES}")

# Add pybind GTSAM module
add_library(ExtrinsicCalibPyModules SHARED src/ExtrinsicCalibPyModules.cpp src/CheckerboardExtrinsicCalibration.cpp src/Cal3DS3.cpp)
target_link_libraries(ExtrinsicCalibPyModules ${catkin_LIBRARIES} ${gtsam_catkin_LIBRARIES})
set_target_properties(
  ExtrinsicCalibPyModules
      PROPERTIES PREFIX
                 ""
                 LIBRARY_OUTPUT_DIRECTORY
                 ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
                 INSTALL_RPATH_USE_LINK_PATH
                 TRUE)


add_executable(extrinsics_calibrator_node src/extrinsics_calibrator_node.cpp)
target_link_libraries(extrinsics_calibrator_node ${catkin_LIBRARIES} ${gtsam_catkin_LIBRARIES} ${OpenCV_LIBRARIES})

catkin_python_setup()

## INSTALL
install(TARGETS extrinsics_calibrator_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(
  TARGETS ExtrinsicCalibPyModules
  LIBRARY
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE)
