cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "flutter_yolo_open_kit")
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

# ONNX Runtime path (downloaded by download_libs.sh)
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs")

# Source files
set(PLUGIN_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/ffi_bridge.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/yolo_detector.cpp"
)

# Create shared library
add_library(${PROJECT_NAME} SHARED ${PLUGIN_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${ONNXRUNTIME_DIR}/include"
    ${OPENCV_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    DART_SHARED_LIB
    __linux__
)

# Link libraries
target_link_directories(${PROJECT_NAME} PRIVATE
    "${ONNXRUNTIME_DIR}/lib"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OPENCV_LIBRARIES}
    onnxruntime
)

# Set output properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "flutter_yolo_open_kit"
    CXX_VISIBILITY_PRESET hidden
    INSTALL_RPATH "$ORIGIN"
)

# For Flutter FFI plugins, set the bundled libraries variable
# This tells Flutter which libraries to bundle with the app
set(flutter_yolo_open_kit_bundled_libraries
    "$<TARGET_FILE:${PROJECT_NAME}>"
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
    PARENT_SCOPE
)
