#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBS_DIR="$SCRIPT_DIR/libs"
INCLUDE_DIR="$SCRIPT_DIR/include"

# ONNX Runtime version
ONNXRUNTIME_VERSION="1.16.3"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    ONNXRUNTIME_ARCH="x64"
elif [ "$ARCH" = "aarch64" ]; then
    ONNXRUNTIME_ARCH="aarch64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz"

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

# Create directories
mkdir -p "$LIBS_DIR/lib"
mkdir -p "$LIBS_DIR/include"

# Download ONNX Runtime if not exists
if [ ! -f "$LIBS_DIR/lib/libonnxruntime.so" ]; then
    echo "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION} for ${ARCH}..."

    ONNXRUNTIME_TGZ="$SCRIPT_DIR/onnxruntime.tgz"
    download_file "$ONNXRUNTIME_URL" "$ONNXRUNTIME_TGZ"

    if [ -f "$ONNXRUNTIME_TGZ" ]; then
        echo "Extracting ONNX Runtime..."
        tar -xzf "$ONNXRUNTIME_TGZ" -C "$SCRIPT_DIR"

        # Move files to libs directory
        EXTRACTED_DIR="$SCRIPT_DIR/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}"
        cp -r "$EXTRACTED_DIR/lib/"* "$LIBS_DIR/lib/"
        cp -r "$EXTRACTED_DIR/include/"* "$LIBS_DIR/include/"

        # Cleanup
        rm -rf "$EXTRACTED_DIR"
        rm -f "$ONNXRUNTIME_TGZ"

        echo "ONNX Runtime installed successfully"
    else
        echo "Error: Failed to download ONNX Runtime"
        exit 1
    fi
else
    echo "ONNX Runtime already exists"
fi

echo ""
echo "========================================"
echo "Linux dependencies ready!"
echo ""
echo "Make sure OpenCV is installed:"
echo "  sudo apt install libopencv-dev"
echo "========================================"
