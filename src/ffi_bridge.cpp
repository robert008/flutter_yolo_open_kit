#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "yolo_detector.hpp"

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Global detector instance
static YoloDetector* g_detector = nullptr;

// Initialize YOLO detector with model path
FFI_PLUGIN_EXPORT int yolo_init(const char* model_path) {
    if (g_detector != nullptr) {
        delete g_detector;
    }
    g_detector = new YoloDetector();
    return g_detector->init(model_path) ? 1 : 0;
}

// Run detection on image file path
// Returns JSON string with detection results (caller must free with free_string)
FFI_PLUGIN_EXPORT char* yolo_detect_path(
    const char* image_path,
    float conf_threshold,
    float iou_threshold
) {
    if (g_detector == nullptr) {
        return strdup("{\"error\":\"Detector not initialized\",\"code\":\"NOT_INITIALIZED\"}");
    }
    return g_detector->detectFromPath(image_path, conf_threshold, iou_threshold);
}

// Run detection on image buffer (BGRA format from camera)
// Returns JSON string with detection results (caller must free with free_string)
FFI_PLUGIN_EXPORT char* yolo_detect_buffer(
    const uint8_t* image_data,
    int width,
    int height,
    int stride,
    float conf_threshold,
    float iou_threshold
) {
    if (g_detector == nullptr) {
        return strdup("{\"error\":\"Detector not initialized\",\"code\":\"NOT_INITIALIZED\"}");
    }
    return g_detector->detectFromBuffer(image_data, width, height, stride, conf_threshold, iou_threshold);
}

// Run detection on YUV420 buffer (Android camera format)
// rotation: 0, 90, 180, 270 degrees clockwise
// Returns JSON string with detection results (caller must free with free_string)
FFI_PLUGIN_EXPORT char* yolo_detect_yuv(
    const uint8_t* y_data,
    const uint8_t* u_data,
    const uint8_t* v_data,
    int width,
    int height,
    int y_row_stride,
    int uv_row_stride,
    int uv_pixel_stride,
    int rotation,
    float conf_threshold,
    float iou_threshold
) {
    if (g_detector == nullptr) {
        return strdup("{\"error\":\"Detector not initialized\",\"code\":\"NOT_INITIALIZED\"}");
    }
    return g_detector->detectFromYUV(
        y_data, u_data, v_data,
        width, height,
        y_row_stride, uv_row_stride, uv_pixel_stride,
        rotation,
        conf_threshold, iou_threshold
    );
}

// Set custom class names (JSON array string)
FFI_PLUGIN_EXPORT void yolo_set_classes(const char* class_names_json) {
    if (g_detector == nullptr) {
        return;
    }

    // Parse simple JSON array: ["class1", "class2", ...]
    std::vector<std::string> names;
    std::string json(class_names_json);

    size_t pos = 0;
    while ((pos = json.find("\"", pos)) != std::string::npos) {
        size_t start = pos + 1;
        size_t end = json.find("\"", start);
        if (end == std::string::npos) break;

        std::string name = json.substr(start, end - start);
        if (!name.empty()) {
            names.push_back(name);
        }
        pos = end + 1;
    }

    if (!names.empty()) {
        g_detector->setClassNames(names);
    }
}

// Release detector resources
FFI_PLUGIN_EXPORT void yolo_release() {
    if (g_detector != nullptr) {
        delete g_detector;
        g_detector = nullptr;
    }
}

// Free allocated string
FFI_PLUGIN_EXPORT void free_string(char* str) {
    if (str != nullptr) {
        free(str);
    }
}

// Get version info
FFI_PLUGIN_EXPORT const char* yolo_get_version() {
    return "0.0.1";
}

// Check if detector is initialized
FFI_PLUGIN_EXPORT int yolo_is_initialized() {
    return (g_detector != nullptr && g_detector->isInitialized()) ? 1 : 0;
}

} // extern "C"
