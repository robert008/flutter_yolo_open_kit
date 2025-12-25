#ifndef FLUTTER_YOLO_OPEN_KIT_H
#define FLUTTER_YOLO_OPEN_KIT_H

#include <stdint.h>

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Initialize YOLO detector with model path
// Returns 1 on success, 0 on failure
FFI_PLUGIN_EXPORT int yolo_init(const char* model_path);

// Run detection on image file path
// Returns JSON string with detection results (caller must free with free_string)
FFI_PLUGIN_EXPORT char* yolo_detect_path(
    const char* image_path,
    float conf_threshold,
    float iou_threshold
);

// Run detection on image buffer (BGRA format from camera)
// Returns JSON string with detection results (caller must free with free_string)
FFI_PLUGIN_EXPORT char* yolo_detect_buffer(
    const uint8_t* image_data,
    int width,
    int height,
    int stride,
    float conf_threshold,
    float iou_threshold
);

// Set custom class names (JSON array string)
FFI_PLUGIN_EXPORT void yolo_set_classes(const char* class_names_json);

// Release detector resources
FFI_PLUGIN_EXPORT void yolo_release(void);

// Free allocated string
FFI_PLUGIN_EXPORT void free_string(char* str);

// Get version info
FFI_PLUGIN_EXPORT const char* yolo_get_version(void);

// Check if detector is initialized
// Returns 1 if initialized, 0 otherwise
FFI_PLUGIN_EXPORT int yolo_is_initialized(void);

#ifdef __cplusplus
}
#endif

#endif // FLUTTER_YOLO_OPEN_KIT_H
