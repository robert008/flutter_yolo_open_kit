#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="v1.0.0"
BASE_URL="https://github.com/robert008/flutter_yolo_open_kit/releases/download/${VERSION}"

# Check if frameworks already exist
OPENCV_DIR="$SCRIPT_DIR/Frameworks/opencv2.framework"
ONNXRUNTIME_LIB="$SCRIPT_DIR/static_libs/libonnxruntime_complete.a"
PLUGIN_LIB="$SCRIPT_DIR/libflutter_yolo_open_kit.a"

download_file() {
    local url=$1
    local output=$2

    echo "Downloading: $url"
    if command -v curl &> /dev/null; then
        curl -L -o "$output" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$output" "$url"
    else
        echo "Error: curl or wget is required"
        exit 1
    fi
}

# Download and extract ios-frameworks.zip if needed
if [ ! -d "$OPENCV_DIR" ] || [ ! -f "$ONNXRUNTIME_LIB" ] || [ ! -f "$PLUGIN_LIB" ]; then
    echo "Downloading iOS frameworks..."
    FRAMEWORKS_ZIP="$SCRIPT_DIR/ios-frameworks.zip"
    download_file "${BASE_URL}/ios-frameworks.zip" "$FRAMEWORKS_ZIP"

    if [ -f "$FRAMEWORKS_ZIP" ]; then
        echo "Extracting frameworks..."
        unzip -q -o "$FRAMEWORKS_ZIP" -d "$SCRIPT_DIR/"
        rm -f "$FRAMEWORKS_ZIP"
        echo "iOS frameworks extracted successfully"
    else
        echo "Error: Failed to download ios-frameworks.zip"
        exit 1
    fi
else
    echo "iOS frameworks already exist"
fi

echo "All iOS dependencies ready!"
