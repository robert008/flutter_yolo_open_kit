#import "YoloKitPlugin.h"

// Declare extern functions to prevent dead code stripping
extern int yolo_init(const char* model_path);
extern char* yolo_detect_path(const char* image_path, float conf_threshold, float iou_threshold);
extern char* yolo_detect_buffer(const uint8_t* image_data, int width, int height, int stride,
                                 float conf_threshold, float iou_threshold);
extern void yolo_set_classes(const char* class_names_json);
extern void yolo_release(void);
extern void free_string(char* str);
extern const char* yolo_get_version(void);
extern int yolo_is_initialized(void);

@implementation YoloKitPlugin

+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
    // FFI plugin - no method channel needed
    // Native functions are called directly via dart:ffi
}

+ (void)load {
    NSLog(@"YoloKit: +load method called");

    // Get version to verify symbols are linked
    volatile const char* version = yolo_get_version();
    NSLog(@"YoloKit: Version: %s", version);

    // Force reference all symbols to prevent linker from stripping them
    if (version == NULL) {
        yolo_init("/nonexistent");
        yolo_detect_path("/nonexistent", 0.0f, 0.0f);
        yolo_detect_buffer(NULL, 0, 0, 0, 0.0f, 0.0f);
        yolo_set_classes("[]");
        yolo_release();
        free_string(NULL);
        yolo_is_initialized();
    }

    NSLog(@"YoloKit: All symbols retained");
}

@end
