cmake_minimum_required(VERSION 2.8.3)

add_executable("${PROJECT_NAME}_simple_rock_sample"
  src/simple_rock_sample.cpp
  src/main.cpp
)
target_link_libraries("${PROJECT_NAME}_simple_rock_sample"
  "${PROJECT_NAME}"
)
install(TARGETS "${PROJECT_NAME}_simple_rock_sample"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)

#TB
set(CMAKE_PREFIX_PATH /home/lab-211/Tali/despot/CPP/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries("${PROJECT_NAME}_simple_rock_sample"
        ${TORCH_LIBRARIES})
#TB
