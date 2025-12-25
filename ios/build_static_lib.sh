#!/bin/bash

# Build static library for iOS
# Usage: ./build_static_lib.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"
OUTPUT_DIR="$SCRIPT_DIR"
BUILD_DIR="$SCRIPT_DIR/build"

# iOS SDK paths
IOS_SDK=$(xcrun --sdk iphoneos --show-sdk-path)
IOS_MIN_VERSION="12.0"

# Include paths
ONNX_HEADERS="$SCRIPT_DIR/Headers"
ONNX_HEADERS_SUB="$SCRIPT_DIR/Headers/onnxruntime"
OPENCV_HEADERS="$SCRIPT_DIR/Frameworks/opencv2.framework/Headers"

# Source files
SOURCES=(
    "$SRC_DIR/yolo_detector.cpp"
    "$SRC_DIR/ffi_bridge.cpp"
)

# Output library name
OUTPUT_LIB="libflutter_yolo_open_kit.a"

echo "Building iOS static library..."
echo "SDK: $IOS_SDK"
echo "Sources: ${SOURCES[*]}"

# Create build directory
mkdir -p "$BUILD_DIR/arm64"

# Compiler flags
CXXFLAGS=(
    -std=c++17
    -O2
    -fPIC
    -fvisibility=hidden
    -DNDEBUG
    -isysroot "$IOS_SDK"
    -arch arm64
    -mios-version-min=$IOS_MIN_VERSION
    -I"$ONNX_HEADERS"
    -I"$ONNX_HEADERS_SUB"
    -I"$OPENCV_HEADERS"
    -I"$SRC_DIR"
)

echo ""
echo "Compiling for arm64 (iOS device)..."

OBJECTS=()
for src in "${SOURCES[@]}"; do
    filename=$(basename "$src" .cpp)
    obj="$BUILD_DIR/arm64/${filename}.o"
    echo "  Compiling: $src -> $obj"
    xcrun clang++ "${CXXFLAGS[@]}" -c "$src" -o "$obj"
    OBJECTS+=("$obj")
done

echo ""
echo "Creating static library..."
xcrun ar rcs "$OUTPUT_DIR/$OUTPUT_LIB" "${OBJECTS[@]}"

echo ""
echo "Library info:"
xcrun ar -t "$OUTPUT_DIR/$OUTPUT_LIB"
ls -lh "$OUTPUT_DIR/$OUTPUT_LIB"

# Verify symbols
echo ""
echo "Exported symbols:"
nm -g "$OUTPUT_DIR/$OUTPUT_LIB" | grep " T " | head -20

# Cleanup
rm -rf "$BUILD_DIR"

echo ""
echo "Build complete: $OUTPUT_DIR/$OUTPUT_LIB"
