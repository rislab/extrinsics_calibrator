cmake_minimum_required(VERSION 2.8.3)
project(extrinsics_calibrator)

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  apriltag_tracker
  gtsam_catkin
)

catkin_package(
  INCLUDE_DIRS
  include 
  CATKIN_DEPENDS
  roscpp
  apriltag_tracker
  gtsam_catkin
  rosbag
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ffast-math -O3 -march=native -Ofast -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -ffast-math -O3 -march=native -Ofast -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -fopenmp -g")

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${gtsam_catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

message("GTSAM_CATKIN Includes : ${gtsam_catkin_INCLUDE_DIRS}")
message("GTSAM_CATKIN Libraries : ${gtsam_catkin_LIBRARIES}")

add_executable(extrinsics_calibrator_node src/extrinsics_calibrator_node.cpp)
target_link_libraries(extrinsics_calibrator_node ${catkin_LIBRARIES} ${gtsam_catkin_LIBRARIES} ${OpenCV_LIBRARIES})

## INSTALL
install(TARGETS extrinsics_calibrator_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE)
