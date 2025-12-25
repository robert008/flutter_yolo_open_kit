cmake_minimum_required(VERSION 3.10)

project(flutter_yolo_open_kit_library VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(flutter_yolo_open_kit SHARED
    ffi_bridge.cpp
    yolo_detector.cpp
)

set_target_properties(flutter_yolo_open_kit PROPERTIES
    PUBLIC_HEADER "flutter_yolo_open_kit.h"
    OUTPUT_NAME "flutter_yolo_open_kit"
    CXX_VISIBILITY_PRESET hidden
)

target_compile_definitions(flutter_yolo_open_kit PUBLIC DART_SHARED_LIB)

# Platform-specific settings
if (ANDROID)
    # jniLibs path
    set(JNILIBS_DIR ${CMAKE_SOURCE_DIR}/../android/src/main/jniLibs/${ANDROID_ABI})

    # Include paths for Android
    target_include_directories(flutter_yolo_open_kit PRIVATE
        ${CMAKE_SOURCE_DIR}/../android/src/main/cpp/include
    )

    # Support Android 15 16k page size
    target_link_options(flutter_yolo_open_kit PRIVATE "-Wl,-z,max-page-size=16384")

    # Link ONNX Runtime
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${JNILIBS_DIR}/libonnxruntime.so)
    target_link_libraries(flutter_yolo_open_kit onnxruntime)

    # Link OpenCV
    add_library(opencv_java4 SHARED IMPORTED)
    set_target_properties(opencv_java4 PROPERTIES IMPORTED_LOCATION ${JNILIBS_DIR}/libopencv_java4.so)
    target_link_libraries(flutter_yolo_open_kit opencv_java4)

    # Link Android log library
    find_library(LOG_LIB log)
    target_link_libraries(flutter_yolo_open_kit ${LOG_LIB})
endif()

if (APPLE)
    # Include paths for iOS
    target_include_directories(flutter_yolo_open_kit PRIVATE
        ${CMAKE_SOURCE_DIR}/../ios/Headers
    )

    # Link frameworks (handled by podspec)
endif()
