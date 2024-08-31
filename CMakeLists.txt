cmake_minimum_required(VERSION 3.5)
project(raisin_bipedal_controller)

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(raisin_controller REQUIRED)
find_package(raisin_interfaces REQUIRED)
find_package(raisin_data_logger REQUIRED)
find_package(raisin_raibo REQUIRED)
find_package(raisim REQUIRED)
find_package(raisin_iekf_plugin REQUIRED)

add_library(${PROJECT_NAME} SHARED
        src/${PROJECT_NAME}/raibo_bipedal_controller.cpp
        )

target_link_libraries(${PROJECT_NAME} raisin_controller::raisin_controller raisim::raisim)
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_srvs
  sensor_msgs
  ament_index_cpp
  raisin_interfaces
  raisin_data_logger
  raisin_raibo
  raisin_iekf_plugin
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

install(
        TARGETS ${PROJECT_NAME}
        EXPORT export_${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)
install(FILES config/params.yaml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rsc DESTINATION lib/..)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
