#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <string>
#include <vector>
#include <memory>

#include <onnxruntime/onnxruntime_cxx_api.h>

struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    float x1, y1, x2, y2;  // bounding box (pixel coordinates)
};

// Model type for different output formats
enum class ModelType {
    YOLOV8,     // [1, 84, 8400] - no objectness
    YOLOX,      // [1, 8400, 85] - has objectness
    PPYOLOE     // [1, N, 6] - already decoded with NMS
};

class YoloDetector {
public:
    YoloDetector();
    ~YoloDetector();

    // Initialize with ONNX model path
    bool init(const std::string& model_path);

    // Set model type explicitly (auto-detected by default)
    void setModelType(ModelType type) { m_model_type = type; }

    // Run detection on image path
    // Returns JSON string (caller must free)
    char* detectFromPath(
        const char* image_path,
        float conf_threshold = 0.25f,
        float iou_threshold = 0.45f
    );

    // Run detection on image buffer (BGRA format from camera)
    char* detectFromBuffer(
        const uint8_t* image_data,
        int width,
        int height,
        int stride,
        float conf_threshold = 0.25f,
        float iou_threshold = 0.45f
    );

    // Run detection on YUV420 buffer (Android camera format)
    // rotation: 0, 90, 180, 270 degrees clockwise
    char* detectFromYUV(
        const uint8_t* y_data,
        const uint8_t* u_data,
        const uint8_t* v_data,
        int width,
        int height,
        int y_row_stride,
        int uv_row_stride,
        int uv_pixel_stride,
        int rotation = 0,
        float conf_threshold = 0.25f,
        float iou_threshold = 0.45f
    );

    // Check if initialized
    bool isInitialized() const { return m_initialized; }

    // Set class names
    void setClassNames(const std::vector<std::string>& names);

    // Release resources
    void release();

private:
    bool m_initialized;
    int m_input_width;
    int m_input_height;
    int m_num_classes;
    ModelType m_model_type;
    std::vector<std::string> m_class_names;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::unique_ptr<Ort::SessionOptions> m_session_options;
    Ort::AllocatorWithDefaultOptions m_allocator;

    std::vector<std::string> m_input_names_str;
    std::vector<std::string> m_output_names_str;

    // Run detection on cv::Mat
    std::vector<Detection> detect(
        const uint8_t* bgr_data,
        int width,
        int height,
        float conf_threshold,
        float iou_threshold
    );

    // Preprocess image for model input (letterbox + normalize)
    std::vector<float> preprocess(
        const uint8_t* bgr_data,
        int width,
        int height,
        float& scale,
        int& pad_x,
        int& pad_y
    );

    // Postprocess model output to detections
    std::vector<Detection> postprocess(
        const float* output,
        const std::vector<int64_t>& output_shape,
        size_t output_count,
        int original_width,
        int original_height,
        float scale,
        int pad_x,
        int pad_y,
        float conf_threshold,
        float iou_threshold
    );

    // Non-maximum suppression
    std::vector<Detection> nms(
        std::vector<Detection>& detections,
        float iou_threshold
    );

    // Calculate IoU between two boxes
    float iou(const Detection& a, const Detection& b);

    // Convert detections to JSON string
    char* toJson(const std::vector<Detection>& detections, long long inference_time_ms, int image_width, int image_height);
};

#endif // YOLO_DETECTOR_HPP
